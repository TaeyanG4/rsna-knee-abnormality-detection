# Definition-aligned report relabelling — prompt v1.1 (FROZEN 2026-10-07 12:27, H-B-28)

v1.1 = v1 plus one clarification added after the 40-report FORMAT pilot (no label outcome and no gold data were looked at): the pilot labeller had to invent a rule for findings mentioned without any grade; v1.1 fixes that rule for every labeller so all batches are treated the same.

Frozen before any output was seen. Do not edit; a changed prompt is a new version (v2) with its own plan item and a reason.

## Task given to each labelling sub-agent

You read knee MRI radiology reports (some are not in English; read them in their own language) and, for each report, decide the 12 findings below **using the competition's official image-read definitions**, as a radiologist following those definitions would label the exam from what the report says. The official labels were made by subspecialty MSK radiologists from the images; borderline findings were graded NEGATIVE to favour specificity; labels are per exam for one knee (if the report covers both knees, label the knee the report identifies as the study knee; if unclear, label the knee described first).

Official definitions:
- ACL: high-grade tear — more than 50% of fibres torn, or complete tear. Low-grade sprain, partial < 50%, mucoid degeneration, old reconstructed graft intact → negative.
- MCL: high-grade acute tear. Sprain, low-grade/partial, chronic thickening, Pellegrini–Stieda without acute tear → negative.
- Medial Meniscus / Lateral Meniscus: a tear (signal reaching the articular surface on at least 2 images). Degeneration/intrasubstance signal without surface extension, post-operative change without a new tear → negative.
- Medial OA / Lateral OA / PF OA (per compartment: medial tibiofemoral, lateral tibiofemoral, patellofemoral): an area of at least about 1 cm with more than 50% loss of cartilage thickness (high-grade/full-thickness chondral loss, moderate-to-severe osteoarthritis with bone-on-bone or deep fissuring/denudation). Mild chondrosis, superficial fibrillation, small focal defects → negative.
- Effusion: moderate or large joint effusion only. Trace/small/physiologic → negative.
- Synovitis: synovitis or synovial thickening/proliferation described as present (including "effusion-synovitis"). If synovitis is not addressed at all, state not_addressed.
- Baker's: popliteal (Baker's) cyst of moderate or large size only. Small cyst → negative.
- Contusion: bone marrow edema/contusion (bone bruise) WITHOUT a fracture line. Degenerative subchondral edema next to cartilage loss is not a contusion unless the report calls it a contusion/bruise.
- Fracture: acute fracture with a cortical break (including osteochondral or avulsion fracture). Old healed fracture, stress reaction without a fracture line → negative.

For each finding output:
- `s`: one of `pos` (meets the definition), `neg` (explicitly absent or present only below the threshold), `unc` (the report is ambiguous about meeting the threshold), `na` (not addressed at all).
- `p`: your probability (0–1) that the official image read is positive, given the report and the definitions (borderline → low).
- Clarification (v1.1): if a thresholded finding is mentioned but its grade, size or severity is not stated at all (e.g. "OA of the PF compartment", "Baker cyst" without size, "ACL tear" without grade), use `s` = `unc` with `p` between 0.4 and 0.6, chosen from the rest of the report (higher when accompanying findings imply severity, lower when they imply a mild process). Use `pos` only when the report indicates the threshold is met.

Output one JSON object per report, one per line: `{"uid": "<StudyInstanceUID>", "f": {"ACL": {"s": "neg", "p": 0.03}, ... all 12 ...}}` with the 12 keys exactly: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture.

## Frozen conversion to training labels (applied by tools, not by the labeller)

- Training label for a finding = `p`, clipped to [0.02, 0.98].
- `na` → 0.05 for every finding except Synovitis; Synovitis `na` → 0.6 × the same report's Effusion `p` (forum evidence that silent synovitis cells follow effusion, D-B-013), clipped as above.
- The 58 gold rows are labelled too but are removed from every training label file; they are used once, after the whole run, as a diagnostic of the extraction (no prompt change is allowed because of it).
