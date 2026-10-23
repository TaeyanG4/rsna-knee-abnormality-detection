# Blank MRI adjudication form — protocol v1.0.0

Planning template only. No real study identifiers, images, labels or report text.
Completed copies belong in the approved private annotation store, not this repo.

Study access token: [assigned privately]
Reader pseudonym / qualified role: [ ]
Manual version: [ ]
Independent first pass / blinded repeat / adjudication: [ ]
Read start / end: [ ]
Target knee and orientation verified: [ ]
Complete available sequences inspected: [ ]
Missing/degraded sequences: [structured code]

For each of ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion and Fracture, record:

| Field | Allowed entry |
|---|---|
| Image finding state | present / absent / indeterminate / not-assessable |
| Severity or grade | Target-manual code; retain ordinal finding if binary mapping unresolved |
| Official target mapping | 1 / 0 / unresolved, with manual clause |
| Evidence location | Private series token, anatomical region and slice span |
| Confidence | high / medium / low, not a probability |
| Limitation / disagreement reason | definition boundary / sequence missing / artifact / equivocal finding / postoperative / target side / other structured code |
| Reader lock | timestamp and record hash |

Do not display model scores, weak labels, reports, heatmaps or selection stratum
in the independent reader screen. A negative requires adequate inspection and
assessability. Effusion must not automatically generate a synovitis label.

Adjudication addendum (after originals are locked):

- Adjudicator's initial blinded finding and time: [ ]
- Original reader disagreement codes: [ ]
- Final state and official mapping: [ ]
- Resolution / remaining uncertainty: [structured code]
- Manual change required: yes / no; if yes, preserve this version and flag the
  complete affected batch for consistent revision, not selected metric fixes.

The software may check completeness and signatures. It must not fill a medical
finding from a model prediction or overwrite an independent reader record.
