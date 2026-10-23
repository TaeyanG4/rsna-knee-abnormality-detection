"""Exact build-time O-bundle removal v1.0.0; P/R224 inference stays untouched."""
import ast

from tools.c85r7_r6w958inf3rco_v1150 import exact


def function(source, name):
    tree = ast.parse(source)
    return ast.get_source_segment(source, next(n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == name))


def drop_preflight(source):
    before = '''    for slug, filename, digest in [(DATASETS[0], "exp169-assets.zip.bin", v5q.A172_ARCHIVE_SHA),
                                   (DATASETS[1], "exp155-assets.zip.bin", EXP155_ARCHIVE_SHA)]:
        p.check_pin(v5q._mount(slug, filename) / filename, digest)
'''
    return exact(source, before, '    # Registered drop_o: no O asset lookup or verification.\n')


P_RECEIPT = '''def candidate_receipt(ids, a, b, payload, codec):
    # Registered P-only parent; O was neither preflighted nor executed.
    return dict(status="COMPLETE_C85R7", version=VERSION, release=RELEASE,
        formula=FORMULA, drop_o=True, rank="average / successful rows; CSV float32",
        failure_policy="unchanged INF3 P survivor substitution; R224 >3% abort gate",
        intended_members=["R", "S_R7", "R224"], executed_members=["R", "S_R7", "R224"],
        skipped_members=["EXP143v2", "EXP155", "Q_A5q"],
        execution_passes=[dict(serving="INF3 unchanged P", arms=["R", "S_R7"]),
                          dict(serving=r224.SOURCE, arms=["R224"])],
        studies=len(ids), findings=12,
        fallbacks=dict(r_prediction=b["r_prediction_fallbacks"],
            s_r7_prediction=b["s_r7_prediction_fallbacks"], r224_prediction=len(b["failures"]["R224"])),
        s_r7_anatomy_anchors=b["s_r7_anatomy_anchors"], r7_identity=b["r7_identity"],
        a172=dict(status="SKIPPED_REGISTERED_DROP_O", executed=False), pr7=b, r224=b["r224"],
        codecs=codec, exposure="EXPOSED", clean_oof=False,
        submission_sha256=hashlib.sha256(payload).hexdigest(), qualified_for_submission=False)'''


def drop_codec(source):
    source = exact(source, 'FORMULA = r224.FORMULA',
        'FORMULA = "INF3 P branch only; R224 retained; registered drop_o"')
    source = exact(source, 'MEMBERS = base.MEMBERS', 'MEMBERS = ["R", "S_R7"]')
    source = exact(source, function(source, 'candidate_receipt'), P_RECEIPT)
    source = exact(source, '    adir.mkdir(); pdir.mkdir()\n', '    pdir.mkdir()\n')
    start = source.index('    for directory in (adir, pdir): write_once')
    end = source.index('    c85.phase(pdir, "child"', start)
    source = exact(source, source[start:end],
        '    write_once(pdir / "ids.json", json_bytes(ids))\n'
        '    # Registered drop_o: no A172 script, assets, child or O phase.\n')
    start = source.index('        ids, exp143, exp155, q, a = c85.authenticated_a172(a172_dir)')
    end = source.index('        need(set(ids) == set(pids)', start)
    source = exact(source, source[start:end], '''        pids, r, ours, b = authenticated_pr7(pr7_dir)
        need(bool(pids), "P-only parent has no study roster")
        ids = pids
        a = None  # O intentionally absent, never authenticated.
        b["source_version"] = "C85R7CODEC " + VERSION
        completed.append("PR7")
''')
    source = exact(source,
        '        values = r224.blend(ids, r, r224_values, ours, exp143, exp155, q, {**a["failures"], **b["failures"]})\n'
        '        expected = r224.reference(r, r224_values, ours, exp143, exp155, q)\n',
        '''        rf = np.asarray([uid in b["failures"]["R"] for uid in ids]) | np.all(r == .5, axis=1)
        f224 = np.asarray([uid in r224_failures for uid in ids])
        sf = np.asarray([uid in b["failures"]["S_R7"] for uid in ids])
        values = r224.p_branch(r, r224_values, ours, rf, f224, sf)
        # Healthy independent transcription, telemetry only (failures use the original policy).
        import pandas as pd
        rank = lambda x: pd.DataFrame(x).rank(method="average", pct=True).to_numpy()
        weight = np.asarray((.5, .5, .5, .5, .5, .2, .6, .5, .8, .35, .5, .5))
        expected = weight * (.65 * rank(r) + .35 * rank(r224_values)) + (1-weight) * rank(ours)
''')
    return source
