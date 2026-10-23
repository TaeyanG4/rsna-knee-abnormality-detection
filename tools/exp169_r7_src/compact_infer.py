"""Compact-model inference worker (Kaggle or local): raw DICOM -> image scores for one list of studies.

Path, identical to how the compact model was trained and evaluated:
  raw DICOM  --mainline training.native_cache.prepare (one study at a time)-->  native uint16 series
             --px_cache.StudySlices.build (the SAME file that built the pixel cache)-->  288 px slices
             --compact_train evaluation windows (every third slice as centre, both views, 3 adjacent slices)-->
             --centre crop 256, fold models of compact_train.Net, mean logit-->  12 probabilities
Nothing is trained here.  No report, no label, no test-time fitting.  Failed or unreached studies are
listed as fallbacks for the controller.

v3 (10-10, R4e): (1) a fold model trained with trainer v9.2+ --syn-decomp (state-dict key syn_beta) is rebuilt with
Net(..., syn_decomp=True) -- needs a pinned compact_train.py whose Net has syn_decomp (v9.3); (2) --crop2: group 2 is built
with its own crop and fed its own centre crop of the same decoded windows; (3) --compile: torch.compile of every fold
model's encoder (mode 'default', dynamic=False: the 10-08 T4 benchmark kaggle/inferbench); before the first study is scored
every encoder batch shape is compiled on synthetic windows (broken-kernel bound) and the first real study runs compiled and
eager: its study-level probabilities must agree within --compile-tol, or else an fp32 forward must show the compiled path
no further from the fp32 model than eager; any error or failed gate puts every model back on its eager encoder (logged as
COMPILE_FALLBACK, recorded in the receipt); (4) --trainer-md5: stop when the imported compact_train.py is not the bundle's
pinned trainer.  Without these flags and without syn_beta in the state dicts the computation is v2's.
v3b (10-10): two groups are mixed in logit space, p = sigmoid((1 - w2) z1 + w2 z2) of the fold-mean logits (v3 mixed probabilities); predictions.csv probabilities are written with %.9g (v3 %.6f rounded ~3 % of cells to 0 / 1).

v4 (10-10, fluid stacker): optional --fluid-stacker <json> (analysis/fit_fluid_stacker.py on the box).  Each study's D2 fluid
volume is measured in the loader worker from the rows px_cache.StudySlices.build has just rendered for the models (no extra
decode; the 10-09 definition of scratch_fluid_1009/fluid_measure.py, see fluid_d2 below), and for every class the json marks
"adopt" the output logit is replaced by the stacker's logit b0 + b_z * z + b_lv * log1p(mL) + b_miss * miss.  A study whose
volume cannot be measured (no fluid-sensitive sagittal series, missing region view, any exception) takes the missing path
(lv = 0, miss = 1), never a crash.
v4b (10-10) = v3b + v4's fluid additions; z = the class's deployed v3b line mix (1 - w2) z1 + w2 z2 of the groups' fold-mean
logits (one group: its fold-mean logit), the same tensor operations as score(); the json must be combine "logit" and its line
weights must equal (1 - w2, w2) of every adopted class (SystemExit otherwise).  Without --fluid-stacker the output is v3b's.
v4c (10-11) = v4b + (1) --flip-tta: the trainer's v9.4 --mirror at inference -- every fold model also scores the mirrored windows
(compact_train_v97.mirror_batch, copied: coronal / axial windows flipped on the width axis, sagittal windows reversed along the slice
axis with rel_pos negated, coronal bilateral views 2 <-> 3), its 12 logits re-indexed by P (Medial <-> Lateral Meniscus, Medial <->
Lateral OA; compact_train_v97.mirror_perm), and the two passes averaged in logit space per fold model; MCL = the plain pass only (the
trainer drops MCL from every mirrored batch); (2) --state-stacker <json> (analysis/fit_state_stacker.py, one model group): for the
classes the json adopts, the output logit is b0 + coef . [z, the class's state-head probabilities, its binaries (, lv, miss)], the
state read-out = compact_train_v97.state_probs of every fold model's forward, averaged over the fold models (the trainer's gold-mean
convention), with --flip-tta averaged over the two passes like the logits (MCL's block and binaries from the plain pass); a class is
replaced by at most one stacker (state or fluid); (3) --dump-state: the per-model state read-out in --dump-logits.  None of them
given = v4b byte for byte.
"""
import argparse, contextlib, csv, hashlib, inspect, io, json, math, os, shutil, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import numpy as np
import torch

UIDK = "StudyInstanceUID"
FORCED_SIDE = {}        # study uid -> "L" / "R": the native render uses this side instead of tags / sagittal IPP x
COMPILE_SANITY_DP = 0.05   # v3 --compile, synthetic windows (random pixels): compiled vs eager |dp| above this = a broken kernel


def compile_gate(d_final, tol, err_c=None, err_e=None):
    """v3 --compile acceptance on the first real study's study-level probabilities.  d_final = max |p_compiled - p_eager|;
    err_c / err_e = max |p - p_fp32| of the compiled / eager path (an fp32 forward of the same study, asked for only when
    d_final > tol).  Returns (True | False | None = needs the fp32 reference, reason)."""
    if d_final <= tol:
        return True, "compiled == eager within %.3g (max |dp| %.3g)" % (tol, d_final)
    if err_c is None or err_e is None:
        return None, "max |dp| compiled vs eager %.3g > %.3g: fp32 reference needed" % (d_final, tol)
    if err_c <= err_e + tol:
        return True, ("compiled vs eager max |dp| %.3g > %.3g, but compiled is no further from the fp32 model than eager "
                      "(max |dp| vs fp32: compiled %.3g, eager %.3g)" % (d_final, tol, err_c, err_e))
    return False, ("compiled vs eager max |dp| %.3g > %.3g and compiled is further from the fp32 model than eager "
                   "(compiled %.3g, eager %.3g)" % (d_final, tol, err_c, err_e))


# ---- v4: D2 fluid volume (training-time definition: scratch_fluid_1009/fluid_measure.py + fluid_io.py on the box, 10-09,
# sagittal part; the same operations in the same order, on the rows StudySlices.build returns instead of the px384_v4 cache rows)
#   series  among the study's Sagittal blocks whose META fluid == 1 (fluid-sensitive), the one with the lowest median of its
#           full-field view (id 0) rows over pixels > 10 (255 when none); ties keep the first in block order
#   region  Effusion: view 4 (100 mm patellofemoral local), Baker's: view 5 (100 mm posterior local) of that series
#   mask    region rows >= 229 (the series' native p99 in the series_companded uint8 mapping), then per slice a morphological
#           opening with an elliptical disk of radius max(1, round(1.5 mm / pixel_mm)) px (keeps pockets >= ~3 mm thick)
#   volume  sum over slices of (mask pixels x pixel_mm^2) x slice spacing / 1000 -> mL; pixel_mm = extent_130 x 130 / 384 per
#           slice row, spacing = expm1(log_spacing) of the series' first row (float32 meta, as the cache stores it)
FLUID_SIZE = 384                     # the definition and the stacker were fitted on the 384 px cache (px384_v4)
FLUID_P99 = 229
FLUID_OPEN_MM = 1.5
FLUID_REGION = {"eff_sag_p99_open": 4, "bak_sag_p99_open": 5}
FLUID_COL = {"log_spacing": 5, "extent_130": 6, "fluid": 7, "fatsat": 8}          # px_cache.META_COLS positions


def _fluid_disk(r):
    import cv2
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fluid_opened(mask, pix_mm):
    import cv2
    out = np.zeros_like(mask)
    for k in range(len(mask)):
        if mask[k].any():
            out[k] = cv2.morphologyEx(mask[k], cv2.MORPH_OPEN, _fluid_disk(FLUID_OPEN_MM / pix_mm[k]))
    return out


def _fluid_ml(mask, pix_mm, spacing):
    return float((mask.reshape(len(mask), -1).sum(1) * pix_mm ** 2).sum() * spacing / 1000.0)


def fluid_d2(px, meta, blocks):
    """D2 volumes (mL) of one study from its rendered rows: px [N, 384, 384] uint8, meta [N, 12] float32, blocks = the
    StudySlices.build blocks ({"plane", "n_slices", "views": [[view_id, first row], ...], "series"}); px may be any object
    whose [a:b] slicing returns those rows.  Returns {column: mL or None, ...}; never raises."""
    rec = {"status": "ok", "eff_sag_p99_open": None, "bak_sag_p99_open": None, "n_sag_fluid_series": 0, "series": None,
           "median": None, "n": None, "spacing": None}
    try:
        best = None
        for b in blocks:
            if b["plane"] != "Sagittal":
                continue
            n = int(b["n_slices"])
            views, first = {}, None
            for v, st in b["views"]:
                m = np.asarray(meta[st])
                if first is None:
                    first = m
                ext = np.asarray(meta[st:st + n, FLUID_COL["extent_130"]], np.float64) * 130.0      # mm spanned by the 384 px side
                views[int(v)] = (int(st), ext / FLUID_SIZE)                                         # per-slice pixel size (mm)
            if first is None or int(first[FLUID_COL["fluid"]]) != 1:
                continue
            rec["n_sag_fluid_series"] += 1
            st0, _ = views[0]
            full = np.asarray(px[st0:st0 + n])
            fg = full[full > 10]
            med = float(np.median(fg)) if fg.size else 255.0
            if best is None or med < best[0]:
                best = (med, b, views, n, float(np.expm1(first[FLUID_COL["log_spacing"]])))
        if best is None:
            rec["status"] = "no fluid-sensitive sagittal series"
            return rec
        med, b, views, n, spacing = best
        rec.update(series=b.get("series"), median=med, n=n, spacing=spacing)
        missing = []
        for col, v in FLUID_REGION.items():
            if v not in views:
                missing.append(col)
                continue
            st, pix = views[v]
            vol = np.asarray(px[st:st + n])
            m99 = (vol >= FLUID_P99).astype(np.uint8)
            rec[col] = _fluid_ml(_fluid_opened(m99, pix), pix, spacing)
        if missing:
            rec["status"] = "no region view for " + ",".join(missing)
    except Exception as exc:                      # the missing path, never a crash
        rec.update(status="error: " + repr(exc)[:200], eff_sag_p99_open=None, bak_sag_p99_open=None)
    return rec


def fluid_stacker_load(path, n_groups):
    """the adopted classes of an analysis/fit_fluid_stacker.py json -> {"classes": {label index: coefficients}, ...}."""
    raw = Path(path).read_bytes()
    st = json.loads(raw.decode("utf-8"))
    feat = st["feature"]
    if feat.get("combine") != "logit":                # v4b: the deployed mix is in logit space (v4 also took "prob")
        raise SystemExit("--fluid-stacker %s: feature.combine %r; v4b mixes the lines in logit space, the stacker's z must be "
                         "combine logit" % (path, feat.get("combine")))
    w = [float(x) for x in feat["weights"]]
    if len(w) != n_groups or len(feat["lines"]) != n_groups:
        raise SystemExit("--fluid-stacker %s was fitted on %d model lines, this run has %d model groups" % (path, len(w), n_groups))
    if abs(sum(w) - 1.0) > 1e-9:
        raise SystemExit("--fluid-stacker %s: line weights %s do not sum to 1" % (path, w))
    cls = {}
    for name, r in st["classes"].items():
        if not r.get("adopt"):
            continue
        if r["volume_column"] not in FLUID_REGION or FLUID_REGION[r["volume_column"]] != int(r["view_id"]):
            raise SystemExit("--fluid-stacker %s: %s uses volume %s / view %s, not a D2 column of this worker" % (
                path, name, r["volume_column"], r["view_id"]))
        co = {k: float(r[k]) for k in ("b0", "b_z", "b_lv", "b_miss")}
        if not all(math.isfinite(v) for v in co.values()):
            raise SystemExit("--fluid-stacker %s: non-finite coefficient for %s" % (path, name))
        wc = [float(x) for x in (feat.get("weights_per_class") or {}).get(name, w)]   # v4b: per-class line weights if the fit had them
        if len(wc) != n_groups or abs(sum(wc) - 1.0) > 1e-9:
            raise SystemExit("--fluid-stacker %s: line weights %s of %s do not fit %d model groups" % (path, wc, name, n_groups))
        cls[name] = dict(co, volume_column=r["volume_column"], weights=wc)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "combine": feat["combine"], "weights": w, "classes": cls,
            "lines": [l.get("run") for l in feat["lines"]], "created": st.get("created")}


def fluid_stack(stk, labels, zmix, fl):
    """stacker logits of the adopted classes for one study.  v4b: zmix = the deployed line mix per class [12] (float64; v3b's
    (1 - w2) z1 + w2 z2 of the groups' fold-mean logits, one group: its fold-mean logit); fl = fluid_d2 record (None = not
    measured).  Returns ({label: z_out}, z_feature [12])."""
    z = np.asarray(zmix, np.float64)
    out = {}
    for name, co in stk["classes"].items():
        v = None if fl is None else fl.get(co["volume_column"])
        miss = v is None or not math.isfinite(float(v)) or float(v) < 0
        lv = 0.0 if miss else float(np.log1p(float(v)))
        x = np.array([z[labels.index(name)], lv, 1.0 if miss else 0.0], np.float64)
        out[name] = co["b0"] + float(x @ np.array([co["b_z"], co["b_lv"], co["b_miss"]], np.float64))
    return out, z


# ---- v4c --flip-tta: the trainer's v9.4 mirror (compact_train_v97.py: MIRROR_PAIRS line 944, mirror_perm lines 947-971,
# mirror_batch lines 973-993), copied; the only difference: no autograd assert (it guards the trainer's evaluation paths)
MIRROR_PAIRS = (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))   # the class pairs P exchanges; the rest fixed


def mirror_perm(labels, spec=None, bin_names=None):
    """compact_train_v97.mirror_perm with LABELS passed in: the class permutation P (its own inverse) and, with a state spec and the
    model's bin_names, the binary permutation; refuses a spec whose paired classes differ in their category lists."""
    swap = {}
    for m_, l_ in MIRROR_PAIRS:
        swap[m_], swap[l_] = l_, m_
    perm = [labels.index(swap.get(l, l)) for l in labels]
    bin_perm = None
    if spec is not None:
        for m_, l_ in MIRROR_PAIRS:
            if spec["categories"][m_] != spec["categories"][l_]:
                raise SystemExit("--flip-tta needs identical state category lists for %s and %s; the spec has %s vs %s" % (
                    m_, l_, spec["categories"][m_], spec["categories"][l_]))
        if bin_names is not None:
            bin_perm = []
            for b, l in bin_names:
                partner = (b, swap.get(l, l))
                if partner not in bin_names:
                    raise SystemExit("--flip-tta: the binary %s exists for %s but not for %s" % (b, l, swap[l]))
                bin_perm.append(bin_names.index(partner))
    return perm, bin_perm


def all_models_v4c(models, models2):
    return list(models) + list(models2)


def mirror_windows(img, meta):
    """compact_train_v97.mirror_batch: img [B, W, K, S, S], meta [B, W, >= 8] with the plane in column 7 (0 sagittal / 1 coronal / 2
    axial).  Coronal and axial windows are flipped on the width axis; a sagittal window is reversed along the slice axis (its K slice
    channels) and its rel_pos (column 2) negated; the coronal bilateral locals change sides: view id 2 <-> 3 (column 1)."""
    assert meta.shape[-1] >= 8, "v4c flip TTA needs the plane column (meta column 7)"
    sag = meta[..., 7].round() == 0
    img = torch.where(sag[:, :, None, None, None], img.flip(2), img.flip(-1))
    meta = meta.clone()
    meta[..., 2] = torch.where(sag, -meta[..., 2], meta[..., 2])
    v = meta[..., 1]
    meta[..., 1] = torch.where(v == 2, torch.full_like(v, 3.), torch.where(v == 3, torch.full_like(v, 2.), v))
    return img, meta


# ---- v4c --state-stacker: the trainer's v9.7 state read-out (compact_train_v97.py: state_layout lines 1139-1157, state_probs 1174-1179)
def state_layout(spec, labels):
    """compact_train_v97.state_layout with LABELS passed in: per class a block of K_c columns (its categories, slots 0 .. K_c-1), then
    one column per binary head (the spec's binaries dict order, each binary's class list order = Net.bin_names)."""
    bad = [l for l in labels if len(spec["prior"][l]) != len(spec["categories"][l])]
    if bad:
        raise SystemExit("--state-stacker: the state spec's prior list length differs from its category count for %s" % bad)
    cols, j = [], 0
    for l in labels:
        k_ = len(spec["categories"][l])
        cols.append({"class": l, "categories": list(spec["categories"][l]), "start": j, "stop": j + k_})
        j += k_
    for b, labs in spec["binaries"].items():
        for l in labs:
            cols.append({"binary": b, "class": l, "column": j})
            j += 1
    return {"columns": cols, "width": j}


def state_readout(model, sl, bl):
    """compact_train_v97.state_probs: per class p = sl.softmax(-1) (the -inf invalid slots get 0) restricted to its K_c valid slots, then
    the sigmoid of every binary logit; float32 [B, D] in the layout of the trainer's oof_state file."""
    p = sl.softmax(-1)
    return torch.cat([p[:, c, :kc] for c, kc in enumerate(model.state_k)] + [torch.sigmoid(bl)], 1).float()


def state_stacker_load(path, labels, layout, spec_md5=None):
    """the adopted classes of an analysis/fit_state_stacker.py json -> per class the feature list, column indices and raw-space
    coefficients; refuses a json whose layout is not this bundle's (state spec + Net binaries) or whose features do not match it, and
    (spec_md5 given) a json fitted on models of another state spec: two specs can share the layout (rr4e and rr5 do) and differ in
    their priors."""
    raw = Path(path).read_bytes()
    st = json.loads(raw.decode("utf-8"))
    if st.get("kind") != "state_stacker_v1":
        raise SystemExit("--state-stacker %s: kind %r, expected state_stacker_v1" % (path, st.get("kind")))
    feat = st["feature"]
    if [float(x) for x in feat.get("weights", [])] != [1.0] or len(feat.get("lines", [])) != 1:
        raise SystemExit("--state-stacker %s: fitted on %s model lines (weights %s); v4c applies it to one model group" % (
            path, len(feat.get("lines", [])), feat.get("weights")))
    if st["layout"]["columns"] != layout["columns"] or int(st["layout"]["width"]) != layout["width"]:
        raise SystemExit("--state-stacker %s: its state layout is not this bundle's (state spec / binaries differ)" % path)
    if spec_md5 is not None and st.get("state_spec_md5") != spec_md5:
        raise SystemExit("--state-stacker %s: fitted on models trained with state spec md5 %s; this bundle's spec has md5 %s" % (
            path, st.get("state_spec_md5"), spec_md5))
    blocks = {c_["class"]: c_ for c_ in layout["columns"] if "categories" in c_}
    bins = [c_ for c_ in layout["columns"] if "binary" in c_]
    cls = {}
    for name, r in st["classes"].items():
        if not r.get("adopt"):
            continue
        if name not in labels:
            raise SystemExit("--state-stacker: unknown class %r" % name)
        b_ = blocks[name]
        want = ["z"] + ["state:" + c for c in b_["categories"]] + ["bin:" + c_["binary"] for c_ in bins if c_["class"] == name]
        cols = [None] + list(range(b_["start"], b_["stop"])) + [c_["column"] for c_ in bins if c_["class"] == name]
        if r.get("volume_column"):
            if r["volume_column"] not in FLUID_REGION:
                raise SystemExit("--state-stacker %s: %s uses volume %s, not a D2 column of this worker" % (path, name, r["volume_column"]))
            want += ["lv", "miss"]
            cols += [None, None]
        keep = [i for i, f in enumerate(want) if f not in set(r.get("dropped") or [])]
        want, cols = [want[i] for i in keep], [cols[i] for i in keep]
        if list(r["features"]) != want:
            raise SystemExit("--state-stacker %s: %s features %s are not the layout's %s" % (path, name, r["features"], want))
        coef = [float(x) for x in r["coef"]]
        if len(coef) != len(want) or not all(math.isfinite(x) for x in coef + [float(r["b0"])]):
            raise SystemExit("--state-stacker %s: %s has %d coefficients for %d features (or non-finite ones)" % (path, name, len(coef), len(want)))
        cls[name] = {"features": want, "columns": cols, "coef": coef, "b0": float(r["b0"]), "volume_column": r.get("volume_column")}
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "classes": cls, "created": st.get("created"),
            "needs_volume": any(v["volume_column"] for v in cls.values()), "lines": feat.get("lines")}


class CStudies(torch.utils.data.Dataset):
    def __init__(self, data, split, uids, tmp, code, size, eval_stride=3, views="v1", slices=3, anchor_weights=None, phys_mm=0.0,
                 side_model=None, side_threshold=2.0, side_max_x=0.0, side_position_mm=20.0, extent_meta=False, fluid=False):
        self.data, self.split, self.uids = str(data), split, list(uids)
        self.fluid = bool(fluid)         # v4: measure the D2 fluid volumes on the rendered rows (--fluid-stacker)
        # extent (M4) fold models read three more meta columns: series index, plane, slices in the series -- the same cache
        # columns (0, 2, 10) compact_train.Studies appends with extent_meta
        self.extent_meta = bool(extent_meta)
        self.tmp, self.code, self.size = str(tmp), str(code), int(size)
        self.eval_stride, self.views, self.slices = int(eval_stride), str(views), int(slices)
        self.phys_mm = float(phys_mm)
        self.anchor_weights = None if anchor_weights is None else str(anchor_weights)
        if self.views in ("v3k", "v4") and not self.anchor_weights:
            raise ValueError("views v3k need --anchor-weights (SKM-TEA checkpoint) to rebuild the knee anchors at inference")
        self.native = None
        # image side check (bundles that ship a side model): studies without one explicit DICOM laterality tag are
        # scored by the orientation classifier on their canonical coronal/axial slices; a confidently mirrored study is
        # re-rendered with the other side (native render, geometry, anchors and views all follow).  None = off.
        self.side_model = None if side_model is None else str(side_model)
        self.side_threshold, self.side_max_x = float(side_threshold), float(side_max_x)
        self.side_position_mm = float(side_position_mm)
        self.side_net = None
        self.source_directory = True     # False: read DICOMs from the chunked competition archive under <data>/download

    def __len__(self):
        return len(self.uids)

    def _setup(self):
        for p in (self.code, str(HERE)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import probe_infer as PI
        self.PI = PI
        self.native = PI.install_native(True)
        import px_cache
        import compact_train
        self.builder = px_cache.StudySlices(self.tmp, [], self.size, self.views)   # (self.slices is the window depth K)
        self.window_rows = compact_train.window_rows          # the trainer's own window list (eval mode)
        self.bank = None
        if self.views in ("v3k", "v4"):
            # knee-centred views: per-series geometry from the DICOM headers (mainline geometry bank, the code that
            # built the training geometry) + SKM-TEA joint anchors (knee_anchor_bank_v2, the code of the training bank)
            import training.geometry_bank as GB
            import knee_anchor_bank_v2 as KB
            self.GB, self.KB = GB, KB
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            if dev == "cpu":
                torch.set_num_threads(2)
            self.bank = KB.AnchorBank(KB.load_model(self.anchor_weights, dev), dev)
        import training.dicom_geometry as G
        if not getattr(G, "_forced_side_wrapped", False):
            import dataclasses
            orig = G.decide_study_side

            def decide(study, tag_values, sagittal_x, _orig=orig):
                r = _orig(study, tag_values, sagittal_x)
                f = FORCED_SIDE.get(str(study))
                return r if f is None else dataclasses.replace(r, side=f, source="image_side_check")
            G.decide_study_side = decide
            G._forced_side_wrapped = True
        if self.side_model:
            import side_common as SC
            self.SC = SC
            sdev = "cuda" if torch.cuda.is_available() else "cpu"
            net = SC.build_side_net()
            net.load_state_dict(torch.load(self.side_model, map_location="cpu", weights_only=True), strict=True)
            self.side_net, self.side_dev = net.eval().to(sdev), sdev

    def _prepare(self, uid, work):
        with contextlib.redirect_stdout(io.StringIO()):
            self.native.prepare(self.data, work, split=self.split, study_ids=[uid], source_directory=self.source_directory)
        if self.bank is not None or self.side_net is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                self.GB.build(root=self.data, manifest=str(work / "manifest.json"), output=str(work / "geometry"),
                              split=self.split, source_directory=self.source_directory)

    def _native_and_geometry(self, uid, work):
        """native render + geometry bank for one study; with a side model, the image side check (may re-render)."""
        self.PI.OneStudySource.current = uid
        self._prepare(uid, work)
        if self.side_net is None:
            return None
        key = hashlib.sha256(uid.encode()).hexdigest()[:24]
        with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
            study = json.load(fh)
        sides = {s.get("side") for s in study["series"]}
        cur = next(iter(sides)) if len(sides) == 1 else None
        status = set()
        for s in study["series"]:
            gp = work / "geometry" / (hashlib.sha256((uid + "/" + s["SeriesInstanceUID"]).encode()).hexdigest()[:24] + ".json")
            if gp.exists():
                status.add(json.loads(gp.read_text(encoding="utf-8")).get("laterality_status"))
        sag = [float(np.median(s["positions"])) for s in study["series"] if s.get("plane") == "Sagittal" and s.get("positions")]
        x = float(np.median(sag)) if sag else None
        note = {"side": cur, "status": "|".join(sorted(str(v) for v in status)), "x": None if x is None else round(x, 1),
                "score": None, "flipped": False}
        if status == {"explicit_study_tags"} and cur in ("L", "R"):
            # one explicit DICOM laterality tag stands unless the knee lies clearly on the other side of the scanner
            # (10-07: 18 training studies had the tag contradicted by both the position and the images)
            pos = None if x is None or abs(x) < self.side_position_mm else ("L" if x >= 0 else "R")
            if pos is None or pos == cur:
                return note
            note["tag_contradicted_by_position"] = True
        if self.side_max_x > 0 and cur in ("L", "R") and x is not None and abs(x) >= self.side_max_x:
            return note                                   # far from the scanner centre: the sagittal IPP x rule holds
        ims = []
        for s in study["series"]:
            if s.get("plane") not in self.SC.SIDE_PLANES:
                continue
            p = Path(s["path"])
            if not p.exists():
                p = work / "series" / os.path.basename(str(s["path"]).replace(chr(92), "/"))
            if p.exists():
                ims.append(self.SC.side_inputs(np.load(p, mmap_mode="r")))
        ims = [v for v in ims if len(v)]
        if not ims:
            return note
        xb = torch.from_numpy(np.concatenate(ims)).float().div_(255.)[:, None]
        with torch.inference_mode():
            z = self.side_net(((xb - 0.45) / 0.25).to(self.side_dev)).float().squeeze(1)
        score = float(z.mean())
        note["score"] = round(score, 3)
        if score < -self.side_threshold:
            new = "R" if cur in (None, "L") else "L"
            FORCED_SIDE[uid] = new
            shutil.rmtree(work, ignore_errors=True)
            self._prepare(uid, work)
            with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
                got = {s.get("side") for s in json.load(fh)["series"]}
            if got != {new}:
                raise ValueError("forced side %s not applied (%s)" % (new, got))
            note.update(flipped=True, new_side=new)
        return note

    def rows(self, uid, forced_side=None):
        """training-cache override builder: the same native -> geometry -> anchors -> views path for one study, with an
        optional forced side; returns (StudySlices.build output, set of sides in the rendered study)."""
        if self.native is None:
            self._setup()
        key = hashlib.sha256(uid.encode()).hexdigest()[:24]
        work = Path(self.tmp) / ("r_" + key[:16])
        if forced_side is not None:
            FORCED_SIDE[uid] = forced_side
        try:
            self.PI.OneStudySource.current = uid
            with contextlib.redirect_stdout(io.StringIO()):
                self.native.prepare(self.data, work, split=self.split, study_ids=[uid], source_directory=self.source_directory)
                self.GB.build(root=self.data, manifest=str(work / "manifest.json"), output=str(work / "geometry"),
                              split=self.split, source_directory=self.source_directory)
            self.builder.cache = str(work)
            with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
                study = json.load(fh)
            if self.bank is not None:
                rec = self.bank.record(uid, study, geo_dir=str(work / "geometry"))
                (work / "anchors").mkdir(exist_ok=True)
                with open(work / "anchors" / (self.KB.study_key(uid) + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
                self.builder.anchors = str(work / "anchors")
            return self.builder.build(uid), {s.get("side") for s in study["series"]}
        finally:
            FORCED_SIDE.pop(uid, None)
            shutil.rmtree(work, ignore_errors=True)

    def __getitem__(self, i):
        uid = self.uids[i]
        t0 = time.perf_counter()
        work = Path(self.tmp) / ("s_" + hashlib.sha256(uid.encode()).hexdigest()[:16])
        t1 = t0
        tf, fl = 0.0, None                                    # v4: fluid measurement seconds / record
        try:
            if self.native is None:
                self._setup()
            side_note = self._native_and_geometry(uid, work)
            t1 = time.perf_counter()
            self.builder.cache = str(work)
            if self.bank is not None:
                key = self.KB.study_key(uid)
                with open(work / "studies" / (key + ".json"), encoding="utf-8") as fh:
                    study = json.load(fh)
                rec = self.bank.record(uid, study, geo_dir=str(work / "geometry"))
                (work / "anchors").mkdir(exist_ok=True)
                with open(work / "anchors" / (key + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh)
                self.builder.anchors = str(work / "anchors")
                anchor_note = {"method": rec["method"], "why": rec["why"][:4]}
            else:
                anchor_note = None
            t2 = time.perf_counter()
            item = self.builder.build(uid)
            px, meta = item["px"].numpy(), item["meta"].numpy()
            if self.fluid:                                    # v4: D2 on the rows just rendered (never raises)
                t3 = time.perf_counter()
                fl = fluid_d2(px, meta, item["blocks"])
                tf = time.perf_counter() - t3
            blocks = [{"n_slices": b["n_slices"], "views": [(0, v, st) for v, st in b["views"]]} for b in item["blocks"]]
            cand = self.window_rows(blocks, lambda f, r: meta[r], "eval", self.slices, self.eval_stride, phys_mm=self.phys_mm)
            if not cand:
                raise ValueError("no usable window")
            rows = np.array(cand, np.int64)                 # (file, centre row, K slice rows)
            img = np.stack([px[rows[:, 2 + d]] for d in range(self.slices)], 1)
            m = meta[rows[:, 1]]
            meta6 = np.stack([m[:, 1], m[:, 3], m[:, 4], m[:, 5], m[:, 6], m[:, 7]], 1).astype(np.float32)
            if self.extent_meta:
                meta6 = np.concatenate([meta6, np.stack([m[:, 0], m[:, 2], m[:, 10]], 1).astype(np.float32)], 1)
            out = {"uid": uid, "error": None, "images": torch.from_numpy(img), "meta": torch.from_numpy(meta6), "anchor": anchor_note,
                   "side": side_note}
        except Exception as exc:
            t2 = locals().get("t2", t1)
            out = {"uid": uid, "error": repr(exc)[:300], "images": torch.zeros((0, self.slices, self.size, self.size), dtype=torch.uint8),
                   "meta": torch.zeros((0, 9 if self.extent_meta else 6), dtype=torch.float32), "anchor": None}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out["seconds"] = [round(t1 - t0, 3), round(time.perf_counter() - t2 - tf, 3), round(t2 - t1, 3)]   # decode, render, anchor
        if self.fluid:
            out["fluid"] = fl
            out["fluid_s"] = round(tf, 4)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--code", required=True)
    ap.add_argument("--models", required=True, help="folder with <arm>_s<seed>_f<fold>.pt state dicts")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--init", default=None, help="ofl_l backbone: the public OFL-L weights (safetensors) the fold models are rebuilt on")
    ap.add_argument("--freeze-blocks", type=int, default=16)
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--views", default="v1", help="px_cache view set the fold models were trained with (v1 / v2)")
    ap.add_argument("--state-spec", default=None, help="state_spec json of the report-state construct the fold models were trained with")
    ap.add_argument("--slices", type=int, default=3, help="adjacent slices per window the fold models were trained with")
    ap.add_argument("--anchor-weights", default=None, help="views v3k: the SKM-TEA checkpoint (anatomy.pt of the code dataset) for the knee anchors")
    ap.add_argument("--phys-mm", type=float, default=0.0, help="window spacing in mm the fold models were trained with (compact_train --phys-mm)")
    ap.add_argument("--eval-stride", type=int, default=3, help="window centres: every k-th slice (3 = training-time validation default; 1 = every slice)")
    ap.add_argument("--assignment", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--amp", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--chunk", type=int, default=None, help="encoder windows per step (default 192; 96 when a CoAtNet group "
                    "is loaded: CoAtNet-2 at 192 does not fit a 16 GB T4)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch", type=int, default=1)
    ap.add_argument("--item-timeout", type=float, default=900.)
    ap.add_argument("--deadline", type=float, default=0.)
    ap.add_argument("--dump-logits", default=None)
    ap.add_argument("--models2", default=None, help="second model group (another architecture), run on the same windows")
    ap.add_argument("--arm2", default=None, help="file prefix of the second group (default: --arm)")
    ap.add_argument("--backbone2", default=None)
    ap.add_argument("--w2", default=None, help="weight of group 2 per class: one number or 12 comma-separated (label order); "
                    "v3b: logit p = (1 - w) z_group1 + w z_group2 (z = fold-mean logit of the group)")
    ap.add_argument("--side-model", default=None, help="orientation classifier (side_common.build_side_net state dict); off when absent")
    ap.add_argument("--side-threshold", type=float, default=2.0, help="flip a study only when its mean logit < -threshold")
    ap.add_argument("--side-max-x", type=float, default=0.0, help="> 0: skip the check for untagged studies this far (mm) from x = 0")
    ap.add_argument("--side-position-mm", type=float, default=20.0, help="a DICOM tag is re-checked only when the knee lies this far on the other side")
    ap.add_argument("--crop2", type=int, default=None, help="v3: centre crop of group 2 (--models2), cut from the same decoded "
                    "windows; group 2 is built with this crop (default: --crop)")
    ap.add_argument("--compile", action="store_true", help="v3: torch.compile every fold model's encoder (checked against eager, "
                    "eager fallback on any error or a difference above --compile-tol)")
    ap.add_argument("--compile-mode", default="default", help="v3: torch.compile mode (inductor backend); 'default' = the 10-08 T4 benchmark")
    ap.add_argument("--compile-backend", default="inductor", help="v3: torch.compile backend; inductor on Kaggle (other values are "
                    "for local plumbing tests on machines without Triton, e.g. aot_eager)")
    ap.add_argument("--compile-tol", type=float, default=1e-3, help="v3: largest accepted |p_compiled - p_eager| of the first real "
                    "study's study-level probabilities; above it the compiled path is kept only when an fp32 forward shows it is no "
                    "further from the fp32 model than eager (+ this tolerance), else every model runs eager")
    ap.add_argument("--trainer-md5", default=None, help="v3: md5 of the bundle's pinned compact_train.py; the worker stops when "
                    "the imported compact_train.py differs")
    ap.add_argument("--fluid-stacker", default=None, help="v4: coefficient json of analysis/fit_fluid_stacker.py; the classes it "
                    "marks adopt get the stacker logit (model logit + D2 fluid volume measured on the rendered rows)")
    ap.add_argument("--fluid-dump", default=None, help="v4: per-study fluid / stacker csv (default <output>/fluid_stacker.csv)")
    ap.add_argument("--flip-tta", action="store_true", help="v4c: every fold model also scores the mirrored windows (the trainer's v9.4 "
                    "--mirror transform); outputs re-indexed (Medial <-> Lateral Meniscus / OA), the two passes averaged in logit space, "
                    "MCL from the plain pass")
    ap.add_argument("--state-stacker", default=None, help="v4c: coefficient json of analysis/fit_state_stacker.py (one model group); the "
                    "classes it adopts get b0 + coef . [z, state probabilities, binaries (, lv, miss)]")
    ap.add_argument("--state-dump", default=None, help="v4c: per-study state-stacker csv (default <output>/state_stacker.csv)")
    ap.add_argument("--dump-state", action="store_true", help="v4c: with --dump-logits, also the per-model state read-out (and the "
                    "mirrored pass with --flip-tta)")
    a = ap.parse_args()
    if a.chunk is None:
        a.chunk = 96 if any("coatnet" in (b or "") for b in (a.backbone, a.backbone2)) else 192
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.tmp or (out / "tmp"))
    tmp.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if a.assignment:
        uids = json.loads(Path(a.assignment).read_text(encoding="utf-8"))
    else:
        with open(Path(a.data) / (a.split + ".csv"), encoding="utf-8") as fh:
            uids = [r[UIDK] for r in csv.DictReader(fh)]
    if len(uids) != len(set(uids)):
        raise SystemExit("duplicate study ids")

    dev = torch.device("cuda")
    for p in (a.code, str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import timm
    import compact_train as CT
    LABELS = CT.LABELS
    trainer_md5 = hashlib.md5(Path(CT.__file__).read_bytes()).hexdigest()       # v3: the trainer module actually imported
    if a.trainer_md5 and trainer_md5 != a.trainer_md5.lower():
        raise SystemExit("compact_train.py at %s has md5 %s, the bundle pins %s" % (CT.__file__, trainer_md5, a.trainer_md5))
    cap = torch.cuda.get_device_capability()[0]
    amp = a.amp if a.amp != "auto" else ("bf16" if cap >= 8 else "fp16")
    state_spec = json.load(open(a.state_spec, encoding="utf-8")) if a.state_spec else None
    crop2 = a.crop if a.crop2 is None else int(a.crop2)                          # v3: group 2's own centre crop
    if a.crop2 is not None and not 0 < crop2 <= a.size:
        raise SystemExit("--crop2 %d: must be in (0, --size %d]" % (crop2, a.size))
    net_has_syn = "syn_decomp" in inspect.signature(CT.Net.__init__).parameters

    def load_group(folder, arm, backbone, crop):
        fs = sorted(Path(folder).glob(arm + "_s*_f*.pt"))
        if not fs:
            raise SystemExit("no model files for arm %s in %s" % (arm, folder))
        ms, n_sp = [], 0
        for p in fs:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            extent = any(k.split(".")[-1] in ("ext_u", "ext_view", "ext_gamma") for k in sd)   # M4 fold model (OA extent)
            if extent:
                N_EXTENT[0] += 1
            spatial = any(k == "sp_q" or k.endswith(".sp_q") for k in sd)       # P2 fold model (per-class spatial attention)
            n_sp += int(spatial)
            # v3: trainer v9.2+ --syn-decomp adds one scalar parameter, Net.syn_beta (z_syn = z_spec + beta * sg(z_eff))
            syn = any(k == "syn_beta" or k.endswith(".syn_beta") for k in sd)
            N_SYN[0] += int(syn)
            kw = {}
            if spatial:
                kw["spatial"] = True
            if extent:
                kw["extent"] = True
            if syn:
                if not net_has_syn:
                    raise SystemExit("%s was trained with --syn-decomp (key syn_beta) but the imported compact_train.py (%s, md5 %s) "
                                     "has no Net(syn_decomp=...): pin trainer v9.3 (compact_train_v93.py)" % (p.name, CT.__file__, trainer_md5))
                kw["syn_decomp"] = True
            m = CT.Net(backbone, a.init if backbone == CT.OFL_BACKBONE else None, freeze_blocks=a.freeze_blocks, crop=crop,
                       state_spec=state_spec, in_chans=a.slices, **kw)
            if backbone == CT.OFL_BACKBONE:
                m.load_deploy_state_dict(sd)          # fold model = public weights + trained tail/head
            else:
                m.load_state_dict(sd, strict=True)
            ms.append(m.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last))
        return fs, ms, n_sp

    N_EXTENT = [0]
    N_SYN = [0]
    files, models, n_spatial = load_group(a.models, a.arm, a.backbone, a.crop)
    n_syn1 = N_SYN[0]
    files2, models2, n_spatial2, W2 = [], [], 0, None
    if a.models2:
        if not a.backbone2 or a.backbone2 == CT.OFL_BACKBONE or a.backbone == CT.OFL_BACKBONE:
            raise SystemExit("--models2 needs --backbone2 (timm backbones only, no OFL shared trunk)")
        files2, models2, n_spatial2 = load_group(a.models2, a.arm2 or a.arm, a.backbone2, crop2)
        w = [float(x) for x in (a.w2 or "0.5").split(",")]
        w = w * len(LABELS) if len(w) == 1 else w
        if len(w) != len(LABELS) or not all(0.0 <= x <= 1.0 for x in w):
            raise SystemExit("--w2: one weight or %d weights in [0, 1]" % len(LABELS))
        W2 = torch.tensor(w, dtype=torch.float64, device=dev)
    shared_trunk = a.backbone == CT.OFL_BACKBONE and len(models) > 1
    if shared_trunk:
        # all fold models share the frozen trunk bit-for-bit; keep one copy and run it once per window batch
        sd0 = models[0].enc.state_dict()
        for m in models[1:]:
            sdm = m.enc.state_dict()
            for k in models[0].enc.frozen_keys:
                if not torch.equal(sdm[k], sd0[k]):
                    raise SystemExit("fold models do not share the frozen trunk: " + k)
    # ---- v4 --fluid-stacker: checked before any study is read (a mismatch is a configuration error, not a per-study failure)
    STK = None
    if a.fluid_stacker:
        if a.size != FLUID_SIZE or a.views != "v4":
            raise SystemExit("--fluid-stacker needs --size %d --views v4 (the D2 definition reads views 0/4/5 of the 384 px rows); "
                             "got --size %d --views %s" % (FLUID_SIZE, a.size, a.views))
        STK = fluid_stacker_load(a.fluid_stacker, 1 + int(bool(models2)))
        for name in STK["classes"]:
            if name not in LABELS:
                raise SystemExit("--fluid-stacker: unknown class %r" % name)
        for name, co in STK["classes"].items():       # v4b: the stacker was fitted on the line mix this bundle deploys
            j = LABELS.index(name)
            want = [1.0 - float(W2[j]), float(W2[j])] if models2 else [1.0]
            if len(co["weights"]) != len(want) or max(abs(x - y) for x, y in zip(co["weights"], want)) > 1e-12:
                raise SystemExit("--fluid-stacker %s: %s was fitted on the line mix %s, this bundle deploys %s (--w2): refit the stacker "
                                 "on the deployed mix or deploy its weights" % (STK["path"], name, co["weights"], want))
        print("FLUID_READY", json.dumps({"json": STK["path"], "sha256": STK["sha256"][:16], "combine": STK["combine"],
                                         "weights": STK["weights"], "adopted": {k: {c: round(v[c], 5) for c in ("b0", "b_z", "b_lv", "b_miss")}
                                                                                 for k, v in STK["classes"].items()},
                                         "volume": {k: v["volume_column"] for k, v in STK["classes"].items()},
                                         "z": "v3b line mix", "weights_per_class": {k: v["weights"] for k, v in STK["classes"].items()}}), flush=True)
    # ---- v4c --flip-tta / --state-stacker / --dump-state: checked before any study is read
    FLIP, SST, AUX = None, None, False
    if a.flip_tta:
        if shared_trunk or a.backbone == CT.OFL_BACKBONE:
            raise SystemExit("--flip-tta: timm backbones only (not the OFL shared trunk)")
        if N_EXTENT[0] == 0:
            raise SystemExit("--flip-tta needs the plane column, which the worker adds to the window meta for extent fold models only")
        if state_spec is not None and any(tuple(m.bin_names) != tuple(models[0].bin_names) for m in all_models_v4c(models, models2)):
            raise SystemExit("--flip-tta: the fold models' binary heads differ")
        perm, bin_perm = mirror_perm(LABELS, state_spec, list(models[0].bin_names) if state_spec is not None else None)
        FLIP = {"perm": perm, "bin_perm": bin_perm, "mcl": LABELS.index("MCL")}
        print("FLIP_TTA_READY", json.dumps({"perm": perm, "mcl_plain_pass_only": True}), flush=True)
    if a.state_stacker or a.dump_state:
        if state_spec is None:
            raise SystemExit("--state-stacker / --dump-state read out the state head: the bundle has no state spec")
        if shared_trunk or a.backbone == CT.OFL_BACKBONE:
            raise SystemExit("--state-stacker / --dump-state: timm backbones only")
        LAYOUT = state_layout(state_spec, LABELS)
        for m in all_models_v4c(models, models2):     # compact_train_v97.state_slots_check: the head's slots are the layout's
            valid = m.state_valid.detach().cpu()
            for c, b_ in enumerate([c_ for c_ in LAYOUT["columns"] if "categories" in c_]):
                kc = b_["stop"] - b_["start"]
                if b_["class"] != LABELS[c] or m.state_k[c] != kc or not bool(valid[c, :kc].all()) or bool(valid[c, kc:].any()):
                    raise SystemExit("--state-stacker: a fold model's state slots for %s are not the spec's %d categories" % (LABELS[c], kc))
            if [(c_["binary"], c_["class"]) for c_ in LAYOUT["columns"] if "binary" in c_] != [tuple(x) for x in m.bin_names]:
                raise SystemExit("--state-stacker: a fold model's binaries are not the spec's")
        AUX = True
    MCL_STATE_COLS = []
    if AUX:
        MCL_STATE_COLS = [i for c_ in LAYOUT["columns"] if c_["class"] == "MCL"
                          for i in (range(c_["start"], c_["stop"]) if "categories" in c_ else [c_["column"]])]
    if a.state_stacker:
        if models2:
            raise SystemExit("--state-stacker: one model group only (the json is fitted on one line's z and state read-out)")
        SST = state_stacker_load(a.state_stacker, LABELS, LAYOUT, hashlib.md5(Path(a.state_spec).read_bytes()).hexdigest())
        both = sorted(set(SST["classes"]) & set(STK["classes"] if STK is not None else []))
        if both:
            raise SystemExit("--state-stacker and --fluid-stacker both adopt %s: a class is replaced by one stacker (fit it jointly)" % both)
        if SST["needs_volume"] and (a.size != FLUID_SIZE or a.views != "v4"):
            raise SystemExit("--state-stacker with volume terms needs --size %d --views v4" % FLUID_SIZE)
        print("STATE_STACKER_READY", json.dumps({"json": SST["path"], "sha256": SST["sha256"][:16], "adopted": {
            k: {"features": len(v["features"]), "volume": v["volume_column"]} for k, v in SST["classes"].items()}}), flush=True)
    print("MODEL_READY", json.dumps({"gpu": torch.cuda.get_device_name(), "amp": amp, "models": len(models),
                                     "models2": len(models2), "spatial_models": n_spatial + n_spatial2, "extent_models": N_EXTENT[0],
                                     "syn_decomp_models": N_SYN[0], "crop": a.crop, "crop2": crop2 if models2 else None,
                                     "compile_requested": bool(a.compile), "trainer_md5": trainer_md5,
                                     "studies": len(uids), "seconds": round(time.time() - t_start, 1)}), flush=True)
    off = (a.size - a.crop) // 2
    off2 = (a.size - crop2) // 2

    def _forward(m, img, meta, mask, mode):
        """one fold model on a study's windows (v2's run_model, unchanged)."""
        if mode == "fp32":
            return m(img, meta, mask, chunk=a.chunk).float()
        with torch.autocast("cuda", dtype=torch.float16 if mode == "fp16" else torch.bfloat16):
            return m(img, meta, mask, chunk=a.chunk).float()

    # ---- v3 --compile: each fold model's encoder through torch.compile; EAGER holds the eager encoder of every model that
    # currently runs a compiled one (empty = the v2 path, bit for bit)
    COMPILE = {"requested": bool(a.compile), "active": False, "mode": a.compile_mode, "backend": a.compile_backend,
               "tol_prob": a.compile_tol, "checked": False, "fallback": None, "shapes": None, "check_seconds": None,
               "decision": None, "study_dprob": None, "fp32_err_compiled": None, "fp32_err_eager": None,
               "max_abs_dlogit": None, "max_abs_dprob": None, "per_group": None, "graphs": None}
    EAGER = {}
    all_models = list(models) + list(models2)

    def set_eager(reason):
        for m in all_models:
            if id(m) in EAGER:
                m.enc = EAGER[id(m)]
        EAGER.clear()
        COMPILE["active"] = False
        COMPILE["fallback"] = str(reason)[:400]
        try:
            import torch._dynamo as _dynamo
            _dynamo.reset()
        except Exception:
            pass
        print("COMPILE_FALLBACK", json.dumps({"reason": COMPILE["fallback"], "every model runs its eager encoder": True}), flush=True)

    @contextlib.contextmanager
    def eager_encoder(m):
        """this fold model on its eager encoder for the duration (the fp32 rescue, the eager side of the compile check)."""
        enc = m.enc
        if id(m) in EAGER:
            m.enc = EAGER[id(m)]
        try:
            yield
        finally:
            if id(m) in EAGER:
                m.enc = enc

    def _forward_aux(m, img, meta, mask, mode):
        """v4c: _forward plus the state / binary logits of the same forward (Net.forward return_aux)."""
        if mode == "fp32":
            z, sl, bl = m(img, meta, mask, chunk=a.chunk, return_aux=True)
        else:
            with torch.autocast("cuda", dtype=torch.float16 if mode == "fp16" else torch.bfloat16):
                z, sl, bl = m(img, meta, mask, chunk=a.chunk, return_aux=True)
        return z.float(), sl.float(), bl.float()

    def run_model_aux(m, img, meta, mask, mode):
        """v4c: run_model with the aux outputs (same compiled / eager handling)."""
        if not EAGER:
            return _forward_aux(m, img, meta, mask, mode)
        if mode == "fp32":
            with eager_encoder(m):
                return _forward_aux(m, img, meta, mask, mode)
        try:
            return _forward_aux(m, img, meta, mask, mode)
        except Exception as exc:
            set_eager("compiled forward failed: " + repr(exc))
            return _forward_aux(m, img, meta, mask, mode)

    def run_model(m, img, meta, mask, mode):
        if not EAGER:                                         # no compiled encoder installed: v2
            return _forward(m, img, meta, mask, mode)
        if mode == "fp32":                                    # the fp32 rescue of a non-finite autocast output stays eager
            with eager_encoder(m):
                return _forward(m, img, meta, mask, mode)
        try:
            return _forward(m, img, meta, mask, mode)
        except Exception as exc:                              # e.g. a recompile that fails: eager from here on, this call again
            set_eager("compiled forward failed: " + repr(exc))
            return _forward(m, img, meta, mask, mode)

    if a.compile:
        if shared_trunk or a.backbone == CT.OFL_BACKBONE:
            COMPILE["fallback"] = "not applied: OFL shared-trunk path (timm backbones only)"
            print("COMPILE_SKIPPED", json.dumps({"reason": COMPILE["fallback"]}), flush=True)
        else:
            try:
                import torch._dynamo as _dynamo
                # every (encoder, batch shape) is one compiled entry; 6 shapes x 8 fold models must never hit the limit (a hit
                # silently runs that frame eagerly: no wrong result, only lost speed)
                for name_, lim_ in (("cache_size_limit", 64), ("recompile_limit", 64), ("accumulated_cache_size_limit", 2048),
                                    ("accumulated_recompile_limit", 2048)):
                    if hasattr(_dynamo.config, name_):
                        setattr(_dynamo.config, name_, max(lim_, int(getattr(_dynamo.config, name_))))
                ckw = dict(dynamic=False, mode=a.compile_mode) if a.compile_backend == "inductor" else dict(dynamic=False, backend=a.compile_backend)
                for m in all_models:
                    EAGER[id(m)] = m.enc
                    m.enc = torch.compile(EAGER[id(m)], **ckw)   # as kaggle/inferbench: m.enc = torch.compile(enc, mode=..., dynamic=False)
                COMPILE["active"] = True
            except Exception as exc:
                set_eager("torch.compile setup failed: " + repr(exc))

    def final_prob(z1, z2):
        """study-level probabilities from per-model logits [n, C], formed exactly as score() forms them."""
        if z2 is None:
            return torch.sigmoid(z1.mean(0).double())
        z = (1 - W2) * z1.mean(0).double() + W2 * z2.mean(0).double()         # v3b: the two groups mixed in logit space
        return torch.sigmoid(z).clamp(1e-12, 1 - 1e-12)

    def compare(zc, ze, gw, key):
        """per-model compiled vs eager: finiteness must agree; the largest |dlogit| / |dp| goes into gw."""
        if not torch.equal(torch.isfinite(zc), torch.isfinite(ze)):
            raise ValueError("compiled and eager outputs differ in finiteness")
        fin = torch.isfinite(zc) & torch.isfinite(ze)
        if bool(fin.any()):
            gw[key + "_dlogit"] = max(gw.get(key + "_dlogit", 0.0), float((zc - ze).abs()[fin].max()))
            gw[key + "_dprob"] = max(gw.get(key + "_dprob", 0.0),
                                     float((torch.sigmoid(zc.double()) - torch.sigmoid(ze.double())).abs()[fin].max()))

    def compile_check(item):
        """Before the first study is scored: (1) compile every encoder batch shape this run can produce (Net.forward pads the
        windows to a multiple of 32 and cuts them into --chunk steps) on synthetic windows, each fold model's compiled output
        against its eager output there only as a broken-kernel bound (finiteness agrees, |dp| <= COMPILE_SANITY_DP); (2) the
        first real study through every fold model compiled and eager: the study-level probabilities (what the submission
        holds) must agree within --compile-tol; if they do not, an fp32 forward of the same study decides -- the compiled path
        is kept only when it is no further from the fp32 model than the eager path we deploy (+ --compile-tol).  Any error or
        a failed gate puts every model back on its eager encoder (COMPILE_FALLBACK)."""
        t0 = time.perf_counter()
        COMPILE["checked"] = True
        try:
            if a.chunk <= 0 or a.chunk % 32:
                raise ValueError("--compile needs --chunk to be a positive multiple of 32 (got %d)" % a.chunk)
            # studies of 32, 64, ..., chunk windows produce every encoder batch a run can see: chunk-sized steps plus a
            # last step of 32 .. chunk windows (Net.forward pads to a multiple of 32)
            sizes = list(range(32, a.chunk + 1, 32))
            gen = torch.Generator().manual_seed(20261010)
            ncol = int(item["meta"].shape[1])
            full = item["images"].to(dev, non_blocking=True)
            meta_r = item["meta"].to(dev)[None]
            mask_r = torch.ones((1, full.shape[0]), dtype=torch.bool, device=dev)
            groups = [("group1", models, a.crop, off)] + ([("group2", models2, crop2, off2)] if models2 else [])
            per_group, real = [], {}
            d_final = err_c = err_e = None
            with torch.inference_mode():
                for gname, ms, cg, og in groups:
                    gw = {"group": gname, "crop": cg}
                    for s in sizes:
                        img_s = torch.randint(0, 256, (1, s, a.slices, cg, cg), generator=gen, dtype=torch.uint8).to(dev)
                        meta_s = torch.zeros((1, s, ncol), dtype=torch.float32, device=dev)
                        mask_s = torch.ones((1, s), dtype=torch.bool, device=dev)
                        for m in ms:
                            zc = _forward(m, img_s, meta_s, mask_s, amp)
                            with eager_encoder(m):
                                ze = _forward(m, img_s, meta_s, mask_s, amp)
                            compare(zc, ze, gw, "synthetic")
                    if not gw.get("synthetic_dprob", 0.0) <= COMPILE_SANITY_DP:
                        raise ValueError("%s synthetic windows: compiled vs eager |dp| %.3g > %.2g (a broken compiled kernel?)" % (
                            gname, gw["synthetic_dprob"], COMPILE_SANITY_DP))
                    img_r = full[None, :, :, og:og + cg, og:og + cg]                     # the first real study
                    zc_l, ze_l = [], []
                    for m in ms:
                        zc = _forward(m, img_r, meta_r, mask_r, amp)[0]
                        with eager_encoder(m):
                            ze = _forward(m, img_r, meta_r, mask_r, amp)[0]
                        compare(zc, ze, gw, "study")
                        zc_l.append(zc)
                        ze_l.append(ze)
                    real[gname] = (torch.stack(zc_l), torch.stack(ze_l), img_r, ms)
                    per_group.append({k: (round(v, 7) if isinstance(v, float) else v) for k, v in gw.items()})
                pc = final_prob(real["group1"][0], real["group2"][0] if models2 else None)
                pe = final_prob(real["group1"][1], real["group2"][1] if models2 else None)
                d_final = float((pc - pe).abs().max())
                ok, why = compile_gate(d_final, a.compile_tol)
                if ok is None:                                # tie-break: an fp32 forward of the same study, eager encoders
                    z32 = {}
                    for gname, (_, _, img_r, ms) in real.items():
                        zz = []
                        for m in ms:
                            with eager_encoder(m):
                                zz.append(_forward(m, img_r, meta_r, mask_r, "fp32")[0])
                        z32[gname] = torch.stack(zz)
                    p32 = final_prob(z32["group1"], z32.get("group2"))
                    err_c, err_e = float((pc - p32).abs().max()), float((pe - p32).abs().max())
                    ok, why = compile_gate(d_final, a.compile_tol, err_c, err_e)
            torch.cuda.synchronize()
            COMPILE.update(shapes=sizes, per_group=per_group, decision=why, study_dprob=round(d_final, 7),
                           fp32_err_compiled=None if err_c is None else round(err_c, 7),
                           fp32_err_eager=None if err_e is None else round(err_e, 7),
                           max_abs_dlogit=max(g.get("study_dlogit", 0.0) for g in per_group),
                           max_abs_dprob=max(g.get("study_dprob", 0.0) for g in per_group))
            try:
                from torch._dynamo.utils import counters as _counters
                COMPILE["graphs"] = int(_counters["stats"]["unique_graphs"])
            except Exception:
                pass
            if not ok:
                set_eager("compile check: " + why)
            else:
                print("COMPILE_READY", json.dumps({k: COMPILE[k] for k in ("mode", "backend", "shapes", "decision", "study_dprob",
                                                                        "fp32_err_compiled", "fp32_err_eager", "per_group", "graphs")}), flush=True)
        except Exception as exc:
            set_eager("compile check failed: " + repr(exc))
        COMPILE["check_seconds"] = round(time.perf_counter() - t0, 1)

    def heads_from_tokens(m, tok, meta, mask):
        """compact_train.Net.head: encoder output -> class logits (the same code the trainer used, state construct included)."""
        return m.head(tok, meta, mask)

    def run_shared(img, meta, mask, mode):
        """OFL fold models: frozen trunk once per window chunk, then every fold's trained tail + head."""
        x = img[mask]
        m0 = models[0]
        tails = [[] for _ in models]
        step = a.chunk or len(x)
        ctx = contextlib.nullcontext() if mode == "fp32" else torch.autocast("cuda", dtype=torch.float16 if mode == "fp16" else torch.bfloat16)
        with ctx:
            for s0 in range(0, len(x), step):
                xs = x[s0:s0 + step].float().div_(255.)
                xs = ((xs - m0.mean) / m0.std).contiguous(memory_format=torch.channels_last)
                tr = m0.enc.trunk(xs)
                for i, m in enumerate(models):
                    tails[i].append(m.enc.tail(tr))
            zs = [heads_from_tokens(m, torch.cat(tails[i]), meta, mask) for i, m in enumerate(models)]
        return torch.stack([z[0].float() for z in zs])

    EXTRA = {}                                                # v4c: per-study side outputs of score_v4c (dump, state stacker)

    def one_pass(m, img, meta, mask):
        """v4c: one fold model's forward with the v4b fp32 rescue; (z [1, 12], state read-out [1, D] or None)."""
        if AUX:
            z, sl, bl = run_model_aux(m, img, meta, mask, amp)
            if not torch.isfinite(z).all():
                z, sl, bl = run_model_aux(m, img, meta, mask, "fp32")
            return z, sl, bl
        z = run_model(m, img, meta, mask, amp)
        if not torch.isfinite(z).all():
            z = run_model(m, img, meta, mask, "fp32")
        return z, None, None

    def score_v4c(ms, img, meta, mask, group=1):
        """v4c: per fold model the plain pass (and with --flip-tta the mirrored pass, re-indexed by P and averaged in logit space, MCL
        from the plain pass); the state read-out of the same forwards (the mirrored one re-indexed by P / bin_perm, averaged, MCL's
        block and binaries from the plain pass).  Returns the per-model logits the study is scored with; EXTRA keeps the rest."""
        out, plain, mirr, st_dep, st_plain, st_mirr = [], [], [], [], [], []
        if FLIP is not None:
            img_m, meta_m = mirror_windows(img, meta)
        for m in ms:
            z, sl, bl = one_pass(m, img, meta, mask)
            st = state_readout(m, sl, bl) if sl is not None else None
            if FLIP is None:
                out.append(z[0])
                if st is not None:
                    st_dep.append(st[0]); st_plain.append(st[0])
                continue
            zm, slm, blm = one_pass(m, img_m, meta_m, mask)
            P = FLIP["perm"]
            zt = 0.5 * (z[0] + zm[0][P])
            zt[FLIP["mcl"]] = z[0][FLIP["mcl"]]
            out.append(zt)
            plain.append(z[0]); mirr.append(zm[0])
            if st is not None:
                stm = state_readout(m, slm[:, P], blm[:, FLIP["bin_perm"]])
                stt = 0.5 * (st[0] + stm[0])
                stt[MCL_STATE_COLS] = st[0][MCL_STATE_COLS]
                st_dep.append(stt); st_plain.append(st[0]); st_mirr.append(state_readout(m, slm, blm)[0])
        EXTRA["group%d" % group] = {"plain": plain, "mirror": mirr, "state": st_dep, "state_plain": st_plain, "state_mirror": st_mirr}
        return out

    @torch.inference_mode()
    def score(images, meta):
        full = images.to(dev, non_blocking=True)
        img = full[None, :, :, off:off + a.crop, off:off + a.crop]
        # v3 --crop2: group 2's own centre crop of the same decoded windows (the same tensor when the crops agree)
        img2 = img if crop2 == a.crop else full[None, :, :, off2:off2 + crop2, off2:off2 + crop2]
        meta = meta.to(dev)[None]
        mask = torch.ones((1, img.shape[1]), dtype=torch.bool, device=dev)
        if shared_trunk:
            zs = run_shared(img, meta, mask, amp)
            if not torch.isfinite(zs).all():
                zs = run_shared(img, meta, mask, "fp32")
            return zs.mean(0), zs
        if FLIP is not None or AUX:                           # v4c: flip TTA / state read-out (v4b's loops below otherwise)
            EXTRA.clear()
            zs = torch.stack(score_v4c(models, img, meta, mask))
            if not models2:
                return zs.mean(0), zs
            zs2 = torch.stack(score_v4c(models2, img2, meta, mask, group=2))
        else:
            zs = []
            for m in models:
                z = run_model(m, img, meta, mask, amp)
                if not torch.isfinite(z).all():
                    z = run_model(m, img, meta, mask, "fp32")
                zs.append(z[0])
            zs = torch.stack(zs)
            if not models2:
                return zs.mean(0), zs
            zs2 = []
            for m in models2:                                 # second architecture on the same windows (one decode)
                z = run_model(m, img2, meta, mask, amp)
                if not torch.isfinite(z).all():
                    z = run_model(m, img2, meta, mask, "fp32")
                zs2.append(z[0])
            zs2 = torch.stack(zs2)
        z = (1 - W2) * zs.mean(0).double() + W2 * zs2.mean(0).double()        # v3b: the two groups mixed in logit space
        pf = torch.sigmoid(z).clamp(1e-12, 1 - 1e-12)
        return torch.log(pf / (1 - pf)), torch.cat([zs, zs2])

    fh = open(out / "predictions.csv", "w", encoding="utf-8", newline="")
    wr = csv.writer(fh)
    wr.writerow([UIDK] + LABELS)
    fallback, stats = {}, {"decode_s": 0., "render_s": 0., "gpu_s": 0., "windows": 0, "wait_s": 0., "rows": 0, "anchor_s": 0.}
    per_study, dump = [], {}
    anchor_methods = {}
    side_counts = {"checked": 0, "flipped": 0}
    # v4: per-study fluid / stacker record
    FL = {"studies": 0, "with_volume": {}, "missing": {}, "status": {}, "fluid_s": 0., "stack_fallback": {}, "first": []}
    if STK is not None:
        FL["with_volume"] = {k: 0 for k in STK["classes"]}
        FL["missing"] = {k: 0 for k in STK["classes"]}
        ffh = open(a.fluid_dump or (out / "fluid_stacker.csv"), "w", encoding="utf-8", newline="")
        fwr = csv.writer(ffh)
        fwr.writerow([UIDK, "status", "n_sag_fluid_series", "series", "median", "n", "spacing", "eff_sag_p99_open", "bak_sag_p99_open",
                      "fluid_s"] + [c + s for c in STK["classes"] for s in ("|z_feature", "|z_out", "|p_model", "|p_out")])

    SS = {"studies": 0, "stack_fallback": {}, "first": []}       # v4c: per-study state-stacker record
    if SST is not None:
        sfh = open(a.state_dump or (out / "state_stacker.csv"), "w", encoding="utf-8", newline="")
        swr = csv.writer(sfh)
        swr.writerow([UIDK] + [c + s for c in SST["classes"] for s in ("|z_feature", "|z_out", "|p_model", "|p_out")])

    def apply_state_stacker(uid, item, zs, p):
        """v4c: overwrite the adopted classes' probabilities with the state stacker's; anything unexpected keeps the model's value."""
        SS["studies"] += 1
        zf = zs.mean(0).double().cpu().numpy()                # the deployed group-1 logit (one group), as score() returns it
        stv = torch.stack(EXTRA["group1"]["state"]).double().mean(0).cpu().numpy()   # arithmetic mean over the fold models
        row = [uid]
        try:
            fl = item.get("fluid")
            for name, co in SST["classes"].items():
                j = LABELS.index(name)
                x = []
                for f, col in zip(co["features"], co["columns"]):
                    if f == "z":
                        x.append(float(zf[j]))
                    elif f in ("lv", "miss"):
                        v = None if fl is None else fl.get(co["volume_column"])
                        miss = v is None or not math.isfinite(float(v)) or float(v) < 0
                        x.append((0.0 if miss else float(np.log1p(float(v)))) if f == "lv" else (1.0 if miss else 0.0))
                    else:
                        x.append(float(stv[col]))
                z_out = co["b0"] + float(np.dot(np.asarray(x, np.float64), np.asarray(co["coef"], np.float64)))
                p_model = float(p[j])
                if math.isfinite(z_out):
                    p[j] = 1.0 / (1.0 + math.exp(-z_out)) if z_out >= 0 else math.exp(z_out) / (1.0 + math.exp(z_out))
                else:
                    SS["stack_fallback"][uid] = "non-finite stacker logit for " + name
                row += ["%.17g" % zf[j], "%.17g" % z_out, "%.17g" % p_model, "%.17g" % float(p[j])]
        except Exception as exc:                              # keep the model's probabilities for this study
            SS["stack_fallback"][uid] = repr(exc)[:200]
            row = [uid] + [""] * (4 * len(SST["classes"]))
        swr.writerow(row)
        sfh.flush()

    def apply_stacker(uid, item, zs, p):
        """v4: overwrite the adopted classes' probabilities with the stacker's; anything unexpected keeps the v3 value."""
        fl = item.get("fluid")
        st = "not measured" if fl is None else fl.get("status", "?")
        FL["studies"] += 1
        FL["status"][st] = FL["status"].get(st, 0) + 1
        FL["fluid_s"] += float(item.get("fluid_s", 0.0))
        n1 = len(models)
        # v4b: z = the deployed v3b line mix, the same tensor operations as score() (one group: its fold-mean logit)
        if models2:
            zmix = ((1 - W2) * zs[:n1].mean(0).double() + W2 * zs[n1:].mean(0).double()).cpu().numpy()
        else:
            zmix = zs.mean(0).double().cpu().numpy()
        row_extra = []
        try:
            zo, zf = fluid_stack(STK, LABELS, zmix, fl)
            for name, z_out in zo.items():
                j = LABELS.index(name)
                col = STK["classes"][name]["volume_column"]
                has = fl is not None and fl.get(col) is not None
                FL["with_volume" if has else "missing"][name] += 1
                p_model = float(p[j])
                if math.isfinite(z_out):
                    p[j] = 1.0 / (1.0 + math.exp(-z_out)) if z_out >= 0 else math.exp(z_out) / (1.0 + math.exp(z_out))
                else:
                    FL["stack_fallback"][uid] = "non-finite stacker logit for " + name
                row_extra += ["%.17g" % zf[j], "%.17g" % z_out, "%.17g" % p_model, "%.17g" % float(p[j])]   # v4b: full precision
        except Exception as exc:                      # keep the v3 probabilities for this study
            FL["stack_fallback"][uid] = repr(exc)[:200]
            row_extra = [""] * (4 * len(STK["classes"]))
        f = fl or {}
        fwr.writerow([uid, st, f.get("n_sag_fluid_series", ""), f.get("series", ""), f.get("median", ""), f.get("n", ""),
                      f.get("spacing", ""), "" if f.get("eff_sag_p99_open") is None else repr(f["eff_sag_p99_open"]),
                      "" if f.get("bak_sag_p99_open") is None else repr(f["bak_sag_p99_open"]), item.get("fluid_s", "")] + row_extra)
        ffh.flush()
        if len(FL["first"]) < 50:
            FL["first"].append([uid, st, f.get("eff_sag_p99_open"), f.get("bak_sag_p99_open"), item.get("fluid_s")])

    def handle(item):
        uid = item["uid"]
        if item["error"] or not len(item["images"]):
            fallback[uid] = item["error"] or "no usable window"
            return
        if EAGER and not COMPILE["checked"]:
            compile_check(item)                       # v3: once, before the first study is scored; not counted in gpu_s
        tg = time.perf_counter()
        try:
            zmean, zs = score(item["images"], item["meta"])
        except Exception as exc:
            fallback[uid] = "gpu: " + repr(exc)[:200]
            return
        if not torch.isfinite(zmean).all():
            fallback[uid] = "non-finite score"
            return
        p = torch.sigmoid(zmean).double().cpu().numpy()
        dt = time.perf_counter() - tg
        if STK is not None:
            ts = time.perf_counter()
            apply_stacker(uid, item, zs, p)           # v4: adopted classes only; the other classes keep v3b's numbers
            FL["stack_s"] = FL.get("stack_s", 0.0) + time.perf_counter() - ts
        if SST is not None:                           # v4c: the state stacker (never the same class as the fluid stacker)
            apply_state_stacker(uid, item, zs, p)
        wr.writerow([uid] + ["%.9g" % float(x) for x in p])
        fh.flush()
        stats["rows"] += 1
        stats["decode_s"] += item["seconds"][0]
        stats["render_s"] += item["seconds"][1]
        stats["anchor_s"] += item["seconds"][2] if len(item["seconds"]) > 2 else 0.
        if item.get("side") and item["side"].get("score") is not None:
            side_counts["checked"] += 1
            side_counts["flipped"] += int(bool(item["side"].get("flipped")))
        if item.get("anchor"):
            anchor_methods[item["anchor"]["method"]] = anchor_methods.get(item["anchor"]["method"], 0) + 1
        stats["gpu_s"] += dt
        stats["windows"] += len(item["images"])
        per_study.append([uid, len(item["images"]), item["seconds"][0], item["seconds"][1], round(dt, 3)])
        if a.dump_logits:
            if FLIP is None and not a.dump_state:
                dump[uid] = zs.float().cpu().numpy().tolist()
            else:                                     # v4c: the scored per-model logits plus the passes / state read-out
                g = EXTRA.get("group1", {})
                rec_ = {"z": zs.float().cpu().numpy().tolist()}
                if FLIP is not None:
                    rec_["z_plain"] = [v.float().cpu().tolist() for v in g.get("plain", [])] + [v.float().cpu().tolist() for v in EXTRA.get("group2", {}).get("plain", [])]
                    rec_["z_mirror_raw"] = [v.float().cpu().tolist() for v in g.get("mirror", [])] + [v.float().cpu().tolist() for v in EXTRA.get("group2", {}).get("mirror", [])]
                if a.dump_state:
                    rec_["state_plain"] = [v.float().cpu().tolist() for v in g.get("state_plain", [])]
                    if FLIP is not None:
                        rec_["state_mirror_raw"] = [v.float().cpu().tolist() for v in g.get("state_mirror", [])]
                dump[uid] = rec_

    pos = 0
    timed_out = False
    while pos < len(uids) and not timed_out:
        ds = CStudies(a.data, a.split, uids[pos:], tmp, a.code, a.size, a.eval_stride, a.views, a.slices, a.anchor_weights, a.phys_mm,
                      a.side_model, a.side_threshold, a.side_max_x, a.side_position_mm, extent_meta=N_EXTENT[0] > 0,
                      fluid=STK is not None or (SST is not None and SST["needs_volume"]))
        loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                                             prefetch_factor=a.prefetch if a.workers else None, persistent_workers=False,
                                             multiprocessing_context="spawn" if a.workers else None,
                                             timeout=a.item_timeout if a.workers else 0)
        try:
            mark = time.perf_counter()
            for item in loader:
                stats["wait_s"] += time.perf_counter() - mark
                handle(item)
                pos += 1
                if pos % 20 == 0 or pos == len(uids) or pos <= 3:
                    el = time.time() - t_start
                    prog = {"done": pos, "of": len(uids), "scored": stats["rows"], "fallback": len(fallback),
                            "elapsed_s": round(el, 1), "s_per_study": round(el / max(pos, 1), 2),
                            "decode_s": round(stats["decode_s"], 1), "render_s": round(stats["render_s"], 1),
                            "gpu_s": round(stats["gpu_s"], 1), "anchor_s": round(stats["anchor_s"], 1),
                            "anchors": anchor_methods, "windows": stats["windows"]}
                    if STK is not None:                   # v4
                        prog.update(fluid_s=round(FL["fluid_s"], 2), fluid_missing=FL["missing"])
                    print(json.dumps(prog), flush=True)
                if a.deadline and time.time() > a.deadline:
                    timed_out = True
                    break
                mark = time.perf_counter()
        except Exception as exc:
            if pos < len(uids):
                fallback[uids[pos]] = "loader: " + repr(exc)[:200]
            pos += 1
        del loader
    for u in uids[pos:]:
        fallback[u] = "not reached before the deadline"
    fh.close()
    if STK is not None:
        ffh.close()
    if SST is not None:
        sfh.close()
    shutil.rmtree(tmp, ignore_errors=True)
    if a.dump_logits:
        Path(a.dump_logits).write_text(json.dumps(dump), encoding="utf-8")
    receipt = {"status": "PASS", "arm": a.arm, "models": [p.name for p in files], "backbone": a.backbone, "shared_trunk": bool(shared_trunk), "init": a.init,
               "spatial_models": n_spatial, "extent_models": N_EXTENT[0], "group2": None if not models2 else {"models": [p.name for p in files2], "backbone": a.backbone2,
                                                                                   "spatial_models": n_spatial2, "w2": W2.tolist(),
                                                                                   "syn_decomp_models": N_SYN[0] - n_syn1, "crop": crop2},
               "syn_decomp_models": n_syn1, "crop": a.crop, "crop2": crop2 if models2 else None, "line_mix": "logit", "compile": COMPILE,
               "trainer_md5": trainer_md5, "trainer_file": str(CT.__file__), "worker_file": Path(__file__).name,
               "studies": len(uids), "scored": stats["rows"], "fallback": fallback, "timed_out": timed_out, "amp": amp,
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "timm": timm.__version__,
               "seconds": round(time.time() - t_start, 1), "stats": {k: round(v, 1) for k, v in stats.items()},
               "per_study": per_study[:50], "anchor_methods": anchor_methods, "views": a.views, "phys_mm": a.phys_mm,
               "side_check": None if not a.side_model else dict(side_counts, threshold=a.side_threshold, max_x=a.side_max_x, position_mm=a.side_position_mm,
                                                                  model_sha256=hashlib.sha256(Path(a.side_model).read_bytes()).hexdigest()),
               "anchor_weights_sha256": (hashlib.sha256(Path(a.anchor_weights).read_bytes()).hexdigest() if a.anchor_weights else None),
               # v3: the worker and trainer entries hash the files actually running / imported (same as HERE/<name> in the notebook)
               "sha256": {n: hashlib.sha256(({"compact_train.py": Path(CT.__file__), "compact_infer.py": Path(__file__).resolve()}.get(n)
                                              or (HERE / n)).read_bytes()).hexdigest()
                          for n in ("px_cache.py", "compact_train.py", "compact_infer.py", "probe_infer.py")
                          + (("side_common.py",) if a.side_model else ())
                          + (("knee_anchor_bank_v2.py",) if a.views in ("v3k", "v4") else ())},
               "model": "compact fine-tuned image model on the pixel cache"}
    if FLIP is not None:                              # v4c
        receipt["flip_tta"] = {"perm": FLIP["perm"], "bin_perm": FLIP["bin_perm"], "mcl": "plain pass only",
                               "transform": "compact_train_v97.mirror_batch (coronal/axial width flip; sagittal slice reversal + rel_pos "
                                            "negated; coronal views 2 <-> 3)", "combine": "per fold model 0.5 (z + P z_mirror), logit space"}
    if SST is not None:                               # v4c
        receipt["state_stacker"] = {"json": SST["path"], "sha256": SST["sha256"], "created": SST["created"], "lines": SST["lines"],
                                    "classes": {k: {"features": v["features"], "b0": v["b0"], "coef": v["coef"], "volume_column": v["volume_column"]}
                                                for k, v in SST["classes"].items()},
                                    "state_combine": "arithmetic mean over the fold models of state_probs" + (" (flip TTA: mean of the two passes)" if FLIP else ""),
                                    "studies": SS["studies"], "stack_fallback": SS["stack_fallback"]}
    if STK is not None:                               # v4
        receipt["fluid_stacker"] = {"json": STK["path"], "sha256": STK["sha256"], "created": STK["created"], "combine": STK["combine"],
                                    "weights": STK["weights"], "lines": STK["lines"], "z": "v3b line mix",
                                    "weights_per_class": {k: v["weights"] for k, v in STK["classes"].items()},
                                    "classes": {k: {c: v[c] for c in ("b0", "b_z", "b_lv", "b_miss", "volume_column")} for k, v in STK["classes"].items()},
                                    "studies": FL["studies"], "with_volume": FL["with_volume"], "missing": FL["missing"], "status": FL["status"],
                                    "fluid_s": round(FL["fluid_s"], 2), "stack_s": round(FL.get("stack_s", 0.0), 3),
                                    "fluid_s_per_study": round(FL["fluid_s"] / max(1, FL["studies"]), 4),
                                    "stack_fallback": FL["stack_fallback"], "first": FL["first"]}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print("WORKER_DONE", json.dumps({k: receipt[k] for k in ("studies", "scored", "timed_out", "seconds", "amp")}),
          "fallback", len(fallback), "compile", json.dumps({k: COMPILE[k] for k in ("requested", "active", "fallback")}), flush=True)


if __name__ == "__main__":
    main()
