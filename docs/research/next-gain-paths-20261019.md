# Prospective gain paths after the fixed v3 joint

Recorded by B / Codex for Taeyang, 2026-10-19. Targets 0.959 and 0.960 are
objectives, not predictions. Solo continuation; no new agent, score query,
download, paid resource, or change to a frozen candidate.

**Superseding audit, 2026-10-19 14:05 KST:** H211 is retained with lower
confidence (priority15 ->14). H212 is removed from the active queue and parked;
the incidence-based continuation below is historical, not an instruction to
resume its census or launch a fit. See the dated audit at the end. Source
observations and original preparation artifacts remain unchanged.

## H211: fixed training augmentation on K1 v3-G

Source audit: the sealed v3-G prepare argv omits `--aug`; the bound trainer
defaults to `crop`. `augment` returns directly after its random crop unless
`--aug v2` is selected. The existing v2 branch applies one rotation/scale and
gamma/contrast transform per study, consistently across windows and adjacent
channels. This is a missing tested mechanism on this particular K1 recipe,
not a claim that all previous models lacked augmentation.

D066 supplies a local reason to test training augmentation, including pooled
five-fold RadR50 benefit; its small partner gain warns that standalone gains
need not transfer. That experiment used another architecture, labels, spatial
contract and augmentation package. Its numbers are not K1 effect estimates.
D220/D233 show that optimizer grouping/checkpoint replacement alone has small,
uncertain mixture effects. The whole current queue and discoveries were searched;
no completed or active exact K1 v3-G augmentation arm was found.

One fixed arm: seed0, fresh pretrained initialization, original3190 training /
977 inference /976 evaluation, original labels, cache, grouping, LR/cosine,
10epochs/finalEMA, clipping and collapse guards. Only `--aug v2 --aug-res 0.0`
changes: rotation +/-10 degrees, scale0.9-1.1, gamma0.8-1.25, contrast0.9-1.1.
No flips, resolution degradation, epoch search or strength sweep. Preserve the
parent finalEMA control and all old source/registrations. Additional RNG draws
are part of this intervention; do not claim identical crop trajectories.
Serving remains identical, so no extra inference pass/member is introduced.

Before launch: finish a separate versioned wrapper/contract, reuse state/loader
repair, check exact source/argument delta, real-cache CPU smoke and bounded GPU
augmentation memory/throughput at the next genuinely free slot. B direct check
is not independent. CPU synthetic augmentation checks do not admit training.
Current H210 is not interrupted. Budget one13-16hour fit plus inference based
on the parent; augmentation overhead is unmeasured. Target completion Oct20
evening if the successor starts early Oct20; no deadline guarantee. Do not start
if the actual finish/qualification window would conflict with the final freeze.

Prospective screen: one paired comparison to seed0 on the same976 using the
existing four-ruler helper, original KEEP versus R7 reported separately.
Report all four macro/target effects and conditional paired intervals. Advance
to a fixed final-mixture check only if mean four-ruler macro delta is positive,
at least3/4 ruler macros are nonnegative and no ruler/target falls below-0.005.
This is an exploratory screen, not a significance or clean-generalization claim.
If it fails, preserve the result; no rescue through target/ratio/epoch changes.
The existing fixed90:10 K3 replacement is the only downstream comparison;
one exact missing serving panel, no adding standalone gains. New hidden execution
must use the validated shared-serving repair and actual RCO COMPLETE.

Seven lenses: fixed report rulers and target tables L1-L4 reused; L5 has seed0
only for the new intervention (H210 is an unaugmented control replicate);
L6 exact joint pending a passing screen; L7 independent image evidence absent,
existing exposed58 only diagnostic. None is silently marked satisfied.

## H212: preserve small series/view groups within the fixed64 training windows

Source audit: `Studies.__getitem__` constructs stride/physical-spacing coverage,
then uniformly samples64 from the combined candidate list. The cap can remove
all representatives of a small series/view group. This is a hypothesis about
finite training allocation, not a proven bug or an efficacy result. It differs
from adding extra series, raising resolution/window count, test-time TTA or the
failed older-family broad-coverage interventions (D006/D125/D134).

First test is a label-free CPU metadata census of the existing3190 training IDs.
Read only completed-cache index and metadata, never pixels/checkpoints or labels.
Freeze one sorted-ID epoch0 draw, seed9000+i, current stride3/physical3.5mm/
slice-jitter0.25/K64. It is a reproducible reference draw, not the historical
training draw order. Record candidate counts and dropped series/view/plane
incidence, aggregate only. If fewer than10% of capped studies lose a group,
park the idea. If at least10% do, prepare one training-only stratified cap:
retain one actual candidate per group, allocate remaining seats proportionally
without replacement, preserve total64 and original candidate order. More than64
groups is an explicit unsupported case requiring a separate design decision.
Do not change study draws, labels, model or serving. This census authorizes no
GPU fit; H211 has the first successor preparation slot. A later fit needs its
own fixed-control rule and budget, and cannot silently combine with H211.

## Existing paths retained

- H210 seed1 addresses replication and supplies another fixed checkpoint;
  it is already running, not a newly discovered architecture or automatic ensemble.
- H202/H204/H207 use completed caches for one exact final-F comparison; they
  resolve joint contribution without more training. H207 retains runtime HOLD.
- H209 is a conditional existing candidate; its tiny mixed D233 result is not
  advertised as the main path to0.960. Existing submitted references remain.
- Distillation, blanket higher resolution/TTA, mirror and public-ratio sweeps
  are not reopened merely to occupy hardware; their adverse/uncertain results stand.

External search connector was unavailable in this session. No new public-source
finding or latest-paper claim is made; these proposals are based on local source
and retained experimental evidence.

## Actual preparation results, 2026-10-19 13:46 KST

H211 source binding and nine CPU synthetic checks passed in
`h211-augmentation/CPU-DESIGN.json`. Tests cover original crop byte parity,
input immutability, shared window/channel transforms, generator-state replay,
active augmentation and actual352px dimensions without CUDA/model loading.
This is CPU_DESIGN_CHECKED_NOT_LAUNCH_READY; full wrapper, real-cache smoke and
the free-slot augmentation memory check remain.

H212 metadata census was deliberately stopped before completion (owned PID17472,
no aggregate output). It made little CPU progress while the training cache was
in use; storage contention is a precaution, not an established diagnosis.
No incidence, gate result or efficacy is inferred. Defer to a free-slot bounded
metadata read, preferably sequential rather than scattered mmap reads. Do not
repeat the unchanged audit alongside training. H210 was never stopped.

## Historical-evidence audit, 2026-10-19 14:05 KST

Requested by Taeyang after the initial proposal. B directly compared the full
relevant discovery entries, including their amendments and reviews, against the
whole current queue. This is a planning correction, not a new experiment or
independent review. Exact-configuration novelty and mechanism novelty are
different: changing the architecture alone does not erase evidence about the
same intervention. Original findings and source artifacts are preserved.

| Path | Historical comparison | Current decision |
|---|---|---|
| H211 K1 training augmentation | D066 supports an augmentation package on RadR50, but its partner contribution was much smaller. D120 is the necessary counterexample: no demonstrated CoAtNet gain in the tested seed/fold, with weak-target losses. D009 has device/software/init confounding; D135 is descriptive and ancestry-limited. No exact K1 v3-G arm was found. | A justified transfer test with uncertain benefit; retain one fixed arm, Confidence2 ->1, priority15 ->14. Do not describe it as a new general mechanism or predict the D066 gain on K1. |
| H212 coverage-preserving K64 | D121 already measured training-bag omissions. D125 and D156 tested CoAtNet coverage, including the second seed and joint readout; D134 tested sampler-only coverage in MIL-T across two seeds. They eliminated omissions but did not establish broad gain. | Mechanism overlap is substantial. Different K1/K64 application is untested, but this alone is insufficient evidence for another expensive fit. Remove from the active queue; no census restart. |
| Existing H202/H204/H207 fixed final-F | D140/D157 support SmallMIL at the partner-surrogate level; D222/D223/D228 supply fixed C4/INF2/joint evidence with adverse folds/targets retained. The previous missing prediction caches have now actually completed and were checked. | Best immediate extension of earlier discoveries: one shared four-arm exact-H189 CPU comparison, owned by these existing items. No new training, duplicate plan item, standalone-gain addition or repeat Q-level analysis. |
| Native320, multi-view and their stack | D178/D184 retain some positive points and target harm. D198's fold3 does not confirm the broad fold2 interpretation; D204 has no resolved stack benefit and lacks a common-fold full-stack comparison. | Keep parked before finals. A smaller final-target harm or independently replicated joint benefit would change this decision; old positive single-fold points do not. |
| S6-all, EXP138/half-O and older teacher/crop routes | D203 has unresolved image-panel ancestry and no resolved gain; D205's selected cell is exploratory; D180/D187/D191/D192 preserve negative or failed routes. | No unchanged reopening, target search or public-score tuning. Existing exact-output/eligibility blockers must actually change first. |

H211 feasibility remains conditional: the existing augmentation implementation
and unchanged serving limit construction and inference cost, but memory and
training overhead at352px are not yet measured. The current13-16hour fit budget
is the parent's observed range, not an augmentation ETA. Prepare the wrapper
without disturbing H210. Launch only if a real free slot and the measured cost
leave time for terminal evaluation and candidate qualification before finals.
Keep the original one-arm screen and report weak-four/all-target outcomes as
well as macro; do not improve the rule after observing results. H210 is an
unaugmented replicate, not replication of the augmentation treatment.

H212's old rule (group omission in >=10% of capped studies) is withdrawn as a
progression gate. It measures occurrence, not the treatment's value, and earlier
experiments already show that eliminating omissions need not help. Reopening
requires a documented K1-specific information-loss mechanism and a fixed test
that distinguishes it from D125/D134/D156, with affordable cost and a concrete
changed decision. A new architecture name, high omission rate, or isolated
target gain alone is insufficient. This does not claim universal ineffectiveness.
The stopped census is neither a completed negative result nor a technical
training defect. Its tool and ATTEMPT.json remain preserved, inactive.

The exact-F continuation must reuse completed H202 Small10 and H204 C4five
outputs plus the authenticated H189 controls. Freeze F0/F2/F4/F24, unchanged
outer formula, ranks, eight-target INF2 mask, UID/target ordering and FP32/CSV
boundaries before computing metrics. Reproduce the original control first,
then compute the two single-change contrasts and joint versus each control
once, using existing paired-statistic helpers. Preserve every target harm and
conditional interval. Do not tune weights or choose a new mask. This is the
first exact-current-final readout, not a rerun of the completed Q-level work.

This audit checked the existing cache result receipts; it did **not** execute
the final-F comparison. Its statistical outcome remains pending. Exposed58 is
small and repeatedly used; even a positive result would not establish clean
private efficacy or remove H207's separate hidden-runtime HOLD. These limits
also prevent treating historical surrogate improvements as a forecast of
0.959 or0.960. No new numerical analysis, GPU launch, submission or resource
claim occurred in this audit.
