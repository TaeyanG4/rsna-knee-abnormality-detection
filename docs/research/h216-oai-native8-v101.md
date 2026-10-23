# H216 native8 side bridge repair v1.0.1

2026-10-19. Original v1 source/registration/ATTEMPT, four completed receipts and
case5 native output remain immutable. v1 stopped at NATIVE_SIDE: the fifth
registered knee is right, but lacks DICOM laterality and sagittal acquisition.
Frozen K1 decides unknown_no_flip. HB88 authenticated this study through the
image03/selection/MOAKS side join and right-sided series descriptions. This is
an input-adapter incompatibility, not evidence of an incorrect clinical label
or a proven cause of old OAI model underperformance.

Use original native8 spec/selection/CPU bounds and unchanged pixel encoder.
The sole semantic repair is an explicit study-side resolver, bound to every
selected eligible HB88 group, requiring consistent authenticated join, source
UID, header-side evidence and any DICOM side tags. Unknown/outside identities,
conflicting tags or unverified groups fail closed. Use the same immutable
geometry's flip_for_plane; do not change source DICOM headers or shared code.
This adapter applies only within this OAI preparation process.

Revalidate/hash-reference the first four completed native studies and their
volumes, without redecoding. Case5's old native output cannot be reused: it
requires correct right-side mirroring. Reuse and byte-verify its existing raw
extraction; decode case5 and unprocessed cases6-8 once into a new directory.
No automatic retry: this is a named reviewed repair after a reproduced defect.
Snapshot includes only code/docs, never identity/label/image files.

Tests reproduce absent-tag/no-sagittal resolution, conflicting DICOM tags,
unregistered identity, correct-side preservation and the production native
identity/frame/metadata checks. B source/test/result review is direct, not
independent. Full-v4 anchors/rendering, paired trainer and efficacy remain HOLD.
