"""H-B-130 per-target aquila-share label arm v1.0.0 (2026-10-13, claude-hb130).

Label-file swap only, through the shared hook runner (training.arms.mil_screen_hooks_v1).
The arm spec equals the shared HB122 control spec except ONE added key, labels.label_swap_v1,
whose registered value names the arm label file (l2_dread_aquila_pertarget_v1, built by
tools/hb130_label_build_v1.py) and pins its file hash, runner semantic hash and the fold-0
positive weights that the unchanged runner derives from it (positive_weights is enforced by
validate_data_with, exactly as EXP184 recomputed them). The arm label CSV/provenance are bound
sources (staged with the run). Every hook is the identity: same windows, AUG1 draws, sampler,
two persistent workers, model, optimizer, schedule, readouts and panel inference path.

The checkpoint's top-level label_set field is written by the shared runner from the
unchanged control labels block; the true arm label set is in provenance.label_set /
provenance.label_swap (the shared runner is hash-bound and is not edited).
Registered by .local/scratch/claude-hb130/PROTOCOL-v1.md.
Changelog 1.0.0: first version.
"""
from training.arms import mil_screen_hooks_v1 as hooks

VERSION = "1.0.0"
PROTOCOL = ".local/scratch/claude-hb130/PROTOCOL-v1.md"
BUILDER = "tools/hb130_label_build_v1.py"
LABEL_NAME = "l2_dread_aquila_pertarget_v1"
LABEL_PATH = "storage/labels/derived_constellation/l2_dread_aquila_pertarget_v1.csv"
LABEL_PROVENANCE = "storage/labels/derived_constellation/l2_dread_aquila_pertarget_v1.provenance.json"
LABEL_SHA256 = "ddef26d3526c7b1cd32289700ee113553f15d1b87e7c345f5c83f21be97700d9"
LABEL_SEMANTIC = "6161500c08ad7826acda88b9b593c75721669b311c6880b3d0a5657b935d747c"
POS_WEIGHT = (4.449650764465332, 9.348213195800781, 1.3974807262420654, 3.895580291748047,
              3.1641364097595215, 5.9647135734558105, 2.501302480697632, 1.7352882623672485,
              2.074486255645752, 5.019646167755127, 3.3302481174468994, 8.444528579711914)
WEAK_FIVE = ("Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis")
SHARES = tuple(60 if t in WEAK_FIVE else 25 for t in hooks.TARGETS)
CONTROL_LABELS = dict(name="l2_dread75_aquila25_v1",
                      path="storage/labels/derived_constellation/l2_dread75_aquila25_v1.csv",
                      csv_sha256="79ce307e246cc54a9a90f32fd1ab21d58ebd23b4b7e7b1a80207adcea1bcc2c9")
LABEL_FIELDS = ("name", "path", "pos_weight", "semantic_sha256")


class LabelShareArm(hooks.Arm):
    name, section, key = "HB130_AQUILA_PERTARGET", "labels", "label_swap_v1"
    screen = "HB130_PERTARGET_AQUILA_V1"
    sources = ("training/arms/hb130_label_share_arm_v1.py", BUILDER, PROTOCOL, LABEL_PATH, LABEL_PROVENANCE)

    def contract(self, seed):
        hooks.need(seed in (42, 43), "unregistered seed")
        return dict(version=VERSION, name=LABEL_NAME, path=LABEL_PATH, csv_sha256=LABEL_SHA256,
                    semantic_sha256=LABEL_SEMANTIC, pos_weight=list(POS_WEIGHT), targets=list(hooks.TARGETS),
                    aquila_share_percent=list(SHARES), builder=BUILDER, provenance=LABEL_PROVENANCE,
                    construction="exp184 mix_rows arithmetic per target: float32((1-s)*Dread + s*aquila_v1)",
                    pos_weight_policy="runner positive_weights on the fold-0 train roster (enforced)",
                    replaces=dict(CONTROL_LABELS), gold58_included=False)

    def validate_recipe(self, spec):
        """single_difference, then the effective labels block used by the runner."""
        c = hooks.single_difference(self, spec)
        d = spec.document
        swap = c[self.section][self.key]
        hooks.need(c["labels"]["name"] == CONTROL_LABELS["name"] and c["labels"]["path"] == CONTROL_LABELS["path"]
                   and d["inputs"]["data"].get(CONTROL_LABELS["path"]) == CONTROL_LABELS["csv_sha256"],
                   "control label binding differs")
        hooks.need(swap["path"] == LABEL_PATH and d["inputs"]["sources"].get(LABEL_PATH) == LABEL_SHA256
                   and d["inputs"]["sources"].get(LABEL_PROVENANCE) is not None, "arm label file binding differs")
        effective = hooks.life.parse_json(hooks.life.canonical_bytes(c))
        effective["labels"] = {field: swap[field] for field in LABEL_FIELDS}
        hooks.need(len(effective["labels"]["pos_weight"]) == 12, "arm positive weights malformed")
        return effective

    def provenance(self, spec):
        return dict(label_set=LABEL_NAME, label_swap=self.contract(spec.document["seed"]),
                    label_swap_note="checkpoint.label_set is the shared runner's control field; see provenance")


ARM = LabelShareArm()
