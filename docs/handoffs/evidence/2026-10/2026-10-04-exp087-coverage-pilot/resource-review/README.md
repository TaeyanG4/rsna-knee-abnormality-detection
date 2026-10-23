# EXP087 v2 scalar resource review

Observed source: `tools/exp087_series_coverage.py:579-625`, last confirmed 2026-10-04 07:32:32 KST. Source snapshot reconstructed verbatim from that tool output. The first attempt to capture it at 07:33:46 failed its v2 assertion because the producer had already changed the file; no runner work occurred. This evidence evaluates the saved v2 only, not the repaired version.

`reproduce_v2.py` compiles only the saved `run()` AST into a fresh namespace. The AST's local `import psutil` is removed and a scalar stand-in is supplied within that namespace. There is no original-module import, `sys.modules` modification, global monkeypatch, real Popen, fixture/header/pixel/cache/model access or pilot execution. Only this evidence script is run by Python.

`stdout.txt` records the two reproduced defects: job.json write failure after fake Popen leaves the fake child unreaped, and the 600-second exit race records PASS at 600.05 seconds. New-source regression requires a separate review.
