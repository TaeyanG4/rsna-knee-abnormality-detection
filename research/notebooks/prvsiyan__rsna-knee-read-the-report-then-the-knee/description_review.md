# Twelve findings, fifty-eight labels, four thousand reports

![Report-supervised knee MRI ensemble](attachment:rsna-knee-report-to-image-hero.jpg)

A knee MRI study here has to be given twelve probabilities — anterior cruciate and medial
collateral ligament injury, medial and lateral meniscal tear, osteoarthritis in each of
the three compartments, effusion, synovitis, Baker's cyst, bone contusion, fracture — and
the score is the unweighted mean of the twelve ROC AUCs.

The decisive fact about this dataset is not in the images. It is in `train.csv`:

| | studies | carry the twelve labels | carry a radiology report |
|---|---|---|---|
| train | 4 407 | **58** | 4 407 |
| test | — | — | **none — there is no `Report` column** |

Fifty-eight labelled studies cannot train an imaging model, and the reports that could
supply the missing four thousand are unavailable at prediction time. So the pipeline is
forced into one shape: **read the reports into targets, fit a pure imaging model against
them, and throw the text away.** Everything downstream — which slices to decode, how large
to make them, which encoder to adapt — is bounded by how well that first step is done.

This notebook is organised in the order the decisions constrain each other.

1. What the score rewards, and what that removes from the design.
2. Where the targets come from — a nine-language report reader, and the two ways to tell
   whether it works when only 58 studies can check it.
3. What the scanner recorded — recovering the acquisition protocol from DICOM headers.
4. Geometry — slice order, physical scale, and which knee was scanned.
5. Reading the pixels once.
6. Turning six views into twelve decisions.
7. Validating without fooling yourself.
8. Three arms, combined the only way the metric permits.

Every analytical number below is computed by the cell above it. Official Kaggle results in
the scorecard are explicitly labelled as external leaderboard evidence.

> **Highest verified public score for this notebook: 0.906.** Version 52 is the
> completed Kaggle row (submission 55399732) that establishes that result; its submitted
> artifact was bitwise matched to the audited 20% RadImageNet rank blend. Version 46
> established 0.899, and Version 49 independently tested the five-fold EfficientNet-B3
> contribution at 0.898. Later guarded candidates remain diagnostic until Kaggle returns
> a completed official score.

| reproducible public run | score | evidence or inference change |
|---|---:|---|
| Pilkwang V15 public anchor | 0.891 | original public 20-member ensemble |
| This notebook V14 | **0.894** | Fracture max + OOF-selected target-wise rank blend |
| This notebook V46 | **0.899** | exact audited specialist artifact; completed Kaggle row |
| This notebook V49 | **0.898** | guarded five-fold EfficientNet-B3 rank contribution |
| This notebook V52 | **0.906** | independent RadImageNet ResNet-50, 20% rank vote; completed Kaggle row 55399732 |
| external Yash Bishnoi reference | **0.903** | inspected five-fold B3 MIL recipe; not this notebook's score |

The scorecard separates completed official rows from local validation and from other
authors' public references. The cells below preserve that boundary: OOF diagnostics can
select or reject an arm, but only a completed Kaggle submission can change the verified
public score above.

If this evidence-first walkthrough saves you time, an upvote helps other competitors
find the reproducible result.


## 1. What the score rewards

$$\text{Score} \;=\; \frac{1}{12}\sum_{i=0}^{11} \mathrm{AUC}_i$$

Three consequences follow directly, and each one removes a design choice rather than
adding one.

**Only order matters.** $\mathrm{AUC}_i$ is invariant under any strictly increasing map of
the scores for label $i$. Calibration is worth nothing here, and so is any fixed
threshold. It also settles how to combine models: averaging raw probabilities lets
whichever arm happens to be most confident dominate, while averaging *ranks* combines the
only information the metric reads. Every combination in §8 is a rank mean.

**Every label costs the same.** Write $M$ for the mean AUC a good model reaches. A label
left at chance contributes $0.5$ instead of roughly $M$, forfeiting

$$\frac{M-0.5}{12}$$

no matter how well the other eleven do. At $M=0.85$ that is $0.029$ — larger than the gap
between neighbouring places on a mature leaderboard. So a rare finding deserves *more*
attention than a common one, because a rare finding is where a model most easily ends up
at chance. That is why §2 spends its effort on the findings the reports mention least,
not on the ones they mention most.

**Prevalence drift is survivable; thresholds are not.** AUC is, in expectation, invariant
to the positive rate, and the data description warns that prevalence need not match across
the training, public and private sets. That would be fatal for an accuracy-like metric. It
is not fatal here — but it does mean one cut has to be named and watched. The report-derived
targets below are graded, not binary; to compute an AUC against them at all they are
binarised at their midpoint for validation only. That cut chooses epochs and arms. It never
reaches a submitted score.

## 2. Where the targets come from

Every training study carries the radiology report written when it was read, and the data
description invites deriving labels from it. The structural fact that decides how is in the
schemas rather than in the prose: `train.csv` has a `Report` column and `test.csv` does
not. Text is available when fitting and absent when predicting. That rules out a fusion
model with a text branch — at inference it would have nothing to read — and leaves the
reports doing two jobs:

1. supplying the training targets, and
2. saying how confidently each one could be read, which becomes a per-finding sample
   weight.

Both matter. A study whose report never mentions synovitis should pull on the synovitis
head far more weakly than one that names it, and a loss that cannot express that trains
every silence as a confident negative.

### Nine languages, one lexicon

The corpus is written in nine languages across three scripts. The extractor is built here
in-line rather than attached as a file, for two reasons: it runs over the whole corpus in
seconds, so there is nothing to save by precomputing; and a notebook that ships its own
labels cannot quietly train against a stale copy of them.

**No language is identified.** Every cue list below carries all nine at once and each
clause is tested against the union. Routing first would mean committing to a guess before
any evidence is read, and the cheap guess fails badly here — `la` is as common in Spanish
as in French, and whichever substring test runs first swallows both. Pooling costs little
in exchange: Greek and Cyrillic cues cannot collide with Latin-script ones at all, and
among the Latin-script languages the vocabulary of interest is close enough that a shared
cue is usually right. The price is paid in coverage instead — a phrasing no listed language
contributes stays unmatched — and coverage is the thing §2.2 measures.

**Normalise, then unwrap, then segment, then scope.** Case, diacritics and separators are
folded first, which also repairs a real codepoint problem: many Greek reports spell mu with
the MICRO SIGN `U+00B5` rather than `U+03BC`, and NFKD maps one onto the other. Then the
lines are unwrapped — see below — then split into clauses, with a heading line attached to
the value beneath it, because a report that reads `Fractures :` and then `Aucune.` states
one thing across two lines.

**Assertion, negation, hedge.** Negation is not an edge case here. For several findings
*most* mentions are negative, because a report lists what was checked and found intact.
Explicit normality counts as absence — *ligamentos cruzados y colaterales dentro de límites
normales* is evidence of absence, not absence of evidence — except where a tear or a high
grade is named in the same breath.

**Grade, do not threshold.** §1 established that only order is read. So a mention is scored
by its emphasis — trace, unqualified, marked — and never binarised.

### 2.1 Four rules that are not obvious

Most of a lexicon is vocabulary, and vocabulary is dull work that pays. Four rules are not
vocabulary, and each of them fixes an error that a term list cannot reach. All four are
behind flags in `FEATURES`, and §2.2 turns them off one at a time and re-measures.

**Unwrap before splitting.** A large share of these reports arrive hard-wrapped at a fixed
column, so a sentence is broken across two lines with no punctuation at the break.
Splitting on newlines then severs a finding from its anatomy:

> `Rotura compleja degenerativa del cuerno anterior, cuerpo, cuerno posterior y parte de la raíz`
> `meniscal posterior del menisco lateral con extrusión asociada.`

The first line says *tear* and names no structure; the second names the lateral meniscus
and says nothing about a tear. Neither clause carries a finding on its own, and the study
comes out silent on a meniscus that the report describes as complexly torn. A line that
does not end in sentence punctuation is a continuation, not a statement.

**Scope negation by direction, and anchor it on the pathology word.** Read at clause scope,

> `Subchondral insufficiency fracture at the medial tibial plateau without articular surface collapse.`

is a denial: it contains *without*. It is in fact an assertion — *without* governs what
follows it, and the fracture precedes it. This is the house style of one of the larger
reporting sites in this corpus, so the error is systematic rather than occasional, and it
falls on `Fracture`, one of the rarest targets. Direction alone is not enough either. In

> `Medial meniscus is not torn.`

the negator stands *after* the noun and before the verb, so scoping the test from the
anatomy match reads the sentence as an assertion. The negator governs the finding, so the
test is anchored on the pathology word — *torn* — not on *meniscus*.

**Let an unlocalised cartilage statement reach all three compartments.** Osteoarthritis is
rarely written as "osteoarthritis". It is written as cartilage loss, chondrosis, a
chondromalacia grade, joint space narrowing, or osteophytes — and often without naming a
compartment at all. `Tricompartmental osteoarthritis`, `gonarthrose`, `Incipient OA of all
three compartmens` are the house styles, not `chondropathy of the medial femorotibial
compartment`. A rule that requires a compartment phrase before it will fire leaves three
quarters of the corpus silent on the three OA targets. So a cartilage statement is
attributed by what else its clause names — *medial* next to a tibiofemoral structure sends
it to the medial compartment, *patella* or *trochlea* to the patellofemoral one — and a
statement that localises to nothing counts for all three at a discount, unless that
compartment was separately and explicitly cleared.

**Read numeric grades on the right scale.** A grade is the most precise thing a knee report
says, and it means opposite things in different places. Grade 3 of a *meniscus* is signal
reaching the articular surface, which is a tear by definition; grades 1 and 2 are
intrasubstance change that is not. A *ligament* runs the other way — grade 1 is a stretch,
grade 2 a partial tear. Folding both into one pathology vocabulary scores a degenerate
meniscus like a torn one and throws away the ordering the metric is built on.

### 2.2 How an extractor like this is validated

This is the part that decides whether any of the above is worth trusting, and it is harder
than writing the rules, because the obvious measurement cannot carry the weight.

**The dangerous failure is silent.** A rule that never fires does not raise an error — it
emits a negative, indistinguishable from a confident one. A lexicon complete in English and
thin in Greek does not look broken; it looks like a corpus in which Greek patients have
fewer findings. And the error is not random: language tracks the reporting institution,
which tracks the scanner and the population, so a gap in one language is a bias aligned with
a site, not noise that averages out.

**Gauge one — agreement, on the 58 annotated studies.** For each target, compare the
extracted score against the annotation and read the AUC. This measures exactly the right
thing and is nearly useless for tuning, because the subset is tiny. The Hanley–McNeil
approximation for the standard error of an AUC $A$ with $n_p$ positives and $n_n$ negatives,

$$\mathrm{SE}(A)=\sqrt{\frac{A(1-A)+(n_p-1)(Q_1-A^{2})+(n_n-1)(Q_2-A^{2})}{n_p n_n}},\qquad
Q_1=\frac{A}{2-A},\quad Q_2=\frac{2A^{2}}{1+A},$$

at $A\approx0.85$ with nine positives among 58 studies lands near $0.083$, so a 95% interval
spans roughly $\pm0.16$. Competitions are decided by differences an order of magnitude
smaller. Choosing between two lexicons on a single one of these numbers is choosing by coin
flip, and it will feel like signal every time. The intervals are drawn below so that this
is visible rather than stated.

**Gauge two — coverage, on all 4 407 reports.** For each (study, target) pair, record
whether *any* rule fired — assertion, negation or hedge. The **silence rate** is the fraction
where none did. It needs no labels, so it runs on the whole corpus rather than on the
annotated handful, and it points straight at missing vocabulary.

| | measures | sample | can decide |
|---|---|---|---|
| agreement | is a fired rule *right* | 58 | whether a target's labels are usable at all |
| silence | does a rule *fire* | 4 407 | which finding to work on next |

Neither substitutes for the other, and the second is the one to steer by. A common finding
that is silent in one language and not another is a lexicon gap. A rare finding that is
silent nearly everywhere is simply rare, and silence there is correct.

**One limit worth naming.** The 58 annotations were read from the images; the reports were
written by a different radiologist on a different day. They disagree — a report saying
*moderate joint effusion* against an annotation of 0, an annotation of Baker's cyst against
a report that calls the study normal. That disagreement is not extractor error and no
lexicon can remove it. It puts a ceiling on gauge one somewhere well below 1.0, which is
another reason to read the intervals rather than the point estimates.

Read that table against §2.2 rather than as a ranking. Three of the five deltas are smaller
than the sampling error of a 58-study AUC, so the honest reading is:

* **`oa_inherit` is the one change large enough to see on gauge one.** It is also the one
  whose effect is visible on gauge two, and the two agree in direction. That is the only
  combination that justifies confidence at this sample size.
* **`unwrap` barely moves agreement and clearly improves coverage.** Kept for the second
  reason. Wrapped lines are a property of how a site exports its reports, not of what those
  reports say, and a rule that repairs the export cannot be wrong about the medicine.
* The rest are kept because the sentence-level argument for them is sound and their
  measured effect is not negative — not because 58 studies said so.

Most of the total quality of these labels is not in this table at all: it is in the
vocabulary, and vocabulary shows up on the silence rate rather than on the ablation. The
findings the reports mention least are the ones §1 says are most expensive to leave at
chance, and they are exactly where the silence bars are tallest — `Fracture`,
`Lateral OA`, `Synovitis`. That is the standing to-do list for this notebook, and it is
readable from a gauge that needs no labels at all.

Two further points about the targets before moving to the pixels.

**Confidence becomes a sample weight.** Every target carries a confidence alongside its
score: high when several clauses agree, low when a finding was named once in passing, and
very low when nothing fired. That number multiplies the per-element loss in §7, so silence
pulls weakly rather than asserting a negative.

**Synovitis is a special case, handled explicitly.** Nearly nine studies in ten never use
the word, and the annotator marks it on close to half. No term list can close that. What a
report does say is whether the joint is wet and irritated — an effusion, a distended bursa,
an inflamed fat pad — and on the silent majority that ordering is the only signal there is.
It enters at low confidence, so it shapes the ranking without asserting a finding. This is
the weakest of the twelve target derivations and the notebook says so rather than hiding it
in an average.

### 2.3 Reading the coverage gauge, which is the point of having it

The silence rate is only useful if it is broken down. Aggregated over the corpus it says
"this target is thin", which anyone could have guessed. Broken down by language it says
*where the vocabulary is missing*, and that is a to-do list.

The two figures below are the ones that actually drove the last several revisions of the
lexicon, so it is worth being precise about how to read them — including the way the left
one can mislead.

**Silence has two causes, and only one of them is a bug.** A target is silent either
because the report never discusses that structure, or because it does and no rule fired.
The first is correct behaviour and no amount of vocabulary fixes it; the second is a gap.
They look identical in the left figure and are separated in the right one, by asking a
question that needs no labels: does the report contain *any* word for this structure at
all?

For the Spanish cruciate cell that split came out 62% never-mentioned against 38%
mentioned-and-missed. The 62% are short reports — a median of 336 characters against 1257
for the ones that do mention it — stating findings and staying silent on everything
normal. Nothing there to fix. The 38% were a genuine gap, and reading a handful of them
showed what it was:

> `Ligamentos cruzados y colaterales sin alteraciones significativas.`

That asserts the cruciates are *normal*. The extractor missed it, because "sin
alteraciones" is a negator plus an abnormality noun — read literally it is a negation, not
a normality phrase, and the rule that recognised normality explicitly refused to fire when
a negator was present. So a ligament a radiologist had looked at and called intact was
scored the same as one the report never mentioned. To a rank metric those are not the
same: explicitly clear must rank *below* unmentioned.

That construction is how most of these languages assert normality — `ohne
Auffälligkeiten`, `geen afwijkingen`, `bez osobitosti`, `χωρίς αλλοιώσεις`, `без
особености` — and adding it dropped the mean silence over all 4 407 studies and twelve
targets from 49.97% to 48.09%, concentrated exactly where predicted: Greek −15 points on
the ligament and meniscus targets, Spanish −6, Bulgarian/Russian −4, Turkish and Croatian
unchanged because they were already covered by a different construction.

That cell is now essentially closed. Spanish studies silent on the cruciate fell from 215
to 141, and of the 141 that remain, **5%** name a cruciate at all — a median report length
of 230 characters says the rest simply do not discuss it. The work moved from the
fixable half of the bar to the half that is not a bug.

The same figure then pointed at Dutch, where `normaal voorkomen menisci` — the bare plural,
naming no side — left both meniscal targets silent. The cruciates and the collaterals had
their plural forms from the start; the menisci had simply been missed.

**What gauge one said about all of this: nothing.** Agreement on the 58 annotated studies
moved by −0.0007 macro AUC, 95% CI [−0.0035, +0.0009] — a flat zero. That is not evidence
the change was worthless. It is evidence that 58 studies, most of them English, cannot see
a Greek and Spanish coverage fix at all. Steering by that number would have rejected the
work; steering by the gauge that runs on all 4 407 found it.

## 3. From report audit to a reusable MRI ensemble

The report-derived label audit above is retained because it documents the weak supervision and its failure modes. The scored path now uses Pilkwang Kim's publicly inspectable 20-checkpoint DINOv2-small ensemble, which reached an official public score of **0.891**. Each member carries its resolution, physical crop, slice ordering, laterality, slot fallback, pooling, and fingerprint contract; any mismatch is fatal rather than silently producing different pixels.

The inference and attached package are adapted with attribution from `pilkwang/rsna-knee-baseline-v1` and `pilkwang/rsna-knee-weights`. The original notebook identity and report analysis remain intact.


## 4. Reading the acquisition


## 3b. What order the slices are in

A series is a directory of files, and the obvious way to walk it is to sort the file names.
That is wrong here, and wrong in a way that produces no error. The file name is the SOP
Instance UID, assigned to be unique rather than ordered, so sorting by it yields a sequence
uncorrelated with anatomy — measured on a series from this corpus, the rank correlation
between file-name order and physical position is $\rho \approx 0.01$. Three things
downstream depend on that order: three adjacent slices as three channels become three
unrelated cross-sections; the middle of the stack becomes a random subset; and reversing
the order to normalise laterality does nothing at all.

The true order is recoverable exactly from geometry every slice carries.
`ImageOrientationPatient` gives the in-plane axes $\hat{r}_x, \hat{r}_y$ and
`ImagePositionPatient` the position $p$ of the first voxel, so

$$\hat{n} \;=\; \hat{r}_x \times \hat{r}_y, \qquad k \;=\; p \cdot \hat{n},$$

and $k$ increases monotonically along the stack. Because $k$ is signed and expressed in
patient coordinates, the stack has a fixed direction along the body's left-right axis —
which is what the laterality normalisation of §5 reverses. `InstanceNumber` is the fallback
where the geometry tags are missing; it usually tracks $k$ up to sign, but interleaved and
multi-echo acquisitions number slices in an order that is not the order they occupy in
space, and it is not signed in patient coordinates.


## 4. Sampling: how many millimetres one pixel is allowed to be

A DICOM slice of $N \times N$ pixels with spacing $s$ mm/pixel covers $Ns$ millimetres of
anatomy. Both vary across this corpus, so a fixed-pixel resize hands the encoder images
whose physical scale differs by a factor of several. But scale normalisation is only half
of it; the other half is a hard limit.

**A feature narrower than two pixels does not survive the resize.** To represent a
structure of width $d$ millimetres the pixel pitch must satisfy $s_{\text{eff}} \le d/2$,
the Nyquist condition applied to the resampling grid. A meniscal tear is one to three
millimetres, so at $d = 1$ mm the pitch must be at most $0.5$ mm — and if it is not, no
capacity downstream recovers the signal, because it was destroyed before the first
convolution. This is a property of the resize, not of the network.

Cropping to a constant physical extent $L$ and resampling to $P$ pixels fixes the pitch:

$$n \;=\; \Big\lfloor \frac{L}{s} \Big\rceil \ \text{pixels}, \qquad
s_{\text{eff}} \;=\; \frac{L}{P}\ \ \text{mm/pixel}, \qquad
\text{token} \;=\; 14\,s_{\text{eff}}\ \ \text{mm}.$$

The crop must be smaller than the smallest field of view or it silently does nothing: if
$L/s$ exceeds the image width the crop cannot be taken and that series passes through
unnormalised, with no error. $L = 130$ mm is below the acquired field of view of almost
every series here while still containing the joint. The resize target then follows from the
tear width rather than convention — at $L = 130$ mm an input of $224$ gives $0.580$
mm/pixel, above the bound for a 1 mm feature, while $336$ gives $0.387$ mm/pixel and puts a
$14$-pixel patch token at $5.4$ mm.

**Intensity needs the same treatment**, MR having no absolute scale. Each series is
normalised to its own 1st and 99th percentile — over the sampled stack rather than per
slice, so slices keep their relative contrast, and percentiles rather than extremes, so one
bright vessel does not compress everything else.


## 5. Normalising left and right

Four of the twelve targets — the two menisci and the medial and lateral tibiofemoral
compartments — are medial/lateral pairs, and a fifth, the medial collateral ligament, is
named for the side it lies on. Medial and lateral are defined relative to the body's
midline, so which side of the *image* they fall on depends on which knee was scanned.
Unless that is normalised, those five labels learn from an axis the model cannot observe.

The correction differs by plane. Coronally and axially the medial-lateral direction lies in
the image plane, so flipping the last axis maps one knee onto the other. Sagittally it is
the *slice* axis: each slice is unchanged by mirroring, and what differs is the order in
which the stack traverses the joint, so the slice order is reversed instead.

`Laterality` is a Type 2C attribute: it may legitimately be absent, and here it is absent
on half the studies — by whole vendors rather than scattered series. Leaving those alone
silently declares them left-sided, so every right knee among them enters the model
mirrored. The patient coordinate system supplies the missing tag: position and orientation
are recorded per image and $+x$ points to the patient's left, so the sign of the image
centre's $x$ says which knee this is.

$$c \;=\; \mathbf{p} \;+\; \mathbf{r}\,\Delta_c \frac{N_c}{2} \;+\; \mathbf{d}\,\Delta_r \frac{N_r}{2},
\qquad \text{side} = \begin{cases} \text{right} & c_x < 0\\ \text{left} & c_x > 0\end{cases}$$

with $\mathbf{p}$ the image position, $\mathbf{r}$ and $\mathbf{d}$ the row and column
direction cosines, and $\Delta$ the pixel spacing. The centre is used rather than
$\mathbf{p}$ itself, which is a corner half a field of view away — enough to change the sign
on a knee scanned near the midline. The median over a study's series is thresholded, and a
study centred within a short distance of the midline is left unresolved rather than guessed,
because inside that band the sign is no better than chance.


## 5b. Reading once, training many times

The cost of this pipeline is dominated by reading, not arithmetic. A study holds several
series and each series tens of slices, so a study is on the order of a hundred and fifty
files. That is affordable once; it is not affordable once per epoch, and fine-tuning needs
the same pixels every epoch. So the slot images are decoded a single time into memory and
held as `uint8`:

$$\text{bytes} \;=\; N_{\text{study}} \times N_{\text{slot}} \times S \times P^{2}$$

with $S$ slices kept per slot at $P$ pixels. The exponent on $P$ is what makes this a real
constraint rather than a detail — the cache grows with the *square* of resolution and only
linearly with slices, so coverage is the cheap axis and resolution the expensive one.

The cached slices form one three-channel encoder input per slot, and the layout generalises
to several such groups per slot: training draws one per step, which doubles as augmentation
along the stack, and inference averages over them. What fixes the number of groups is a
budget rather than a capacity — the cache is allowed a fraction of the memory reported free,
deliberately below the whole of it, since it is the one allocation large enough that
overshooting ends the run rather than slowing it, and the encoder, its activations and the
buffers in flight come from the same pool. The size compared against that budget is the sum
of *both* caches, training and test, because both are resident at once.

A valid submission file is written before any of this begins and overwritten once real
predictions exist, so the run always leaves a scoreable file behind.


## 6. Aggregating slots into twelve decisions

A study arrives as up to six slot embeddings $x_s \in \mathbb{R}^{d}$ with a presence mask
$m_s \in \{0,1\}$. Pooling them identically would discard the reason the protocol has three
planes: each finding is read on particular sequences, and a mean over slots dilutes the one
carrying the evidence with five that do not.

Project each slot, add a learned slot identity, give every diagnosis $o$ its own query
$q_o \in \mathbb{R}^{H}$, and let it attend over the slots with absent ones masked out of
the softmax:

$$h_s \;=\; \phi(x_s) + e_s, \qquad
\alpha_{o,s} \;=\; \frac{\exp\!\big(\langle h_s, q_o\rangle / \sqrt{H}\big)\, m_s}
{\sum_{s'} \exp\!\big(\langle h_{s'}, q_o\rangle / \sqrt{H}\big)\, m_{s'}},$$

$$c_o \;=\; \sum_s \alpha_{o,s}\, h_s, \qquad
\ell_o \;=\; \langle c_o, w_o \rangle + b_o .$$

The masked softmax renormalises over whatever the study actually contains, so a missing
axial series shifts a diagnosis's attention onto the sequences that are present instead of
feeding it a zero vector.

**The head is deliberately this small.** The label is attached to the *study*, so nothing
in the supervision says which part of a study carries the finding, and a richer aggregation
would have no signal to learn that from. Where the supervision is coarse, so is the head.


### Why the encoder is trained rather than frozen

A frozen self-supervised encoder is bounded by something no work downstream can reach.
Resolution, encoder size, slice coverage and slot aggregation change how much the model
looks and how closely, but none changes the vocabulary it looks *with*, so every one of those
axes runs into the same ceiling — and that ceiling should be expected to bind here, since the
encoder learned its features from natural images, where nothing resembles the signal a torn
meniscus makes on a proton-density sequence.

So the encoder is adapted, with two restraints. **Only the last blocks move** — early blocks
of a vision transformer are generic edge and texture filters, late blocks are where
semantics live, and there may not be enough supervision here to improve the early ones while
there is certainly enough to damage them. **The encoder learns far more slowly than the
head** — the head is random at initialisation and has everything to learn, the encoder starts
from a good solution and needs only to be moved off it, so a single learning rate would
either leave the head untrained or destroy the encoder in the first few hundred steps.

Targets remain the report-derived labels of §2, weighted by the per-finding confidence that
comes with them, so a report that never mentions a finding pulls weakly on that output
rather than asserting a negative there. The studies carrying per-condition annotations are
weighted above every derived row, since they are the only labels read from the images.


## 6b. Reading weights instead of learning them

Nothing requires the scored run to be the run that learned the weights. A notebook may
attach a dataset, weights are a dataset, and the part that genuinely cannot be done in
advance is the part depending on studies nobody has seen — reading them, and predicting.
Learning here instead caps the model at what one accelerator fits inside the time limit,
and spends that time again on every submission for a result that does not change. So when
a package of trained members is attached this section reads it; when none is, the sections
below train one.

**Members, not a model.** Each member carries the preprocessing it was fitted on. Two
members fitted at different resolutions cannot share a decode; two fitted alike can. So
members are grouped by the pixels they need, each group is decoded once, and a member added
later joins the list unchanged.

What groups them is more than the resolution and the crop. Four further decisions settle
what a slice *is* — which one comes next along the stack, which knees are mirrored, whether
a slot may be filled from a sequence that does not match its predicate, and what stands in
for a slice that will not decode — and none of them changes a single shape. A member fitted
under one reading and decoded under another therefore loads cleanly, runs, and returns a
submission computed from the wrong image, so the reading travels with the member.

**A member must prove it is the model it was.** Loading a state dictionary succeeds
whenever the shapes line up, and shapes line up across every difference that matters — a
changed normalisation, resize, or slice band. None of those raise. So each member carries
the answer it gave to a seeded question, recomputed before use; a mismatch stops the run
rather than being averaged into a submission.

**How a member is read.** Training draws one group of consecutive slices per step. At
inference, looking more costs no extra decoding — the cache already holds $S$ slices, so one
group from the middle is $1$ forward pass, the disjoint groups are $S/G$, and every
consecutive run of $G$ slices is $S-G+1$. Where the average is taken is free but not
neutral: averaging logits then squashing is a geometric mean of odds, averaging
probabilities an arithmetic mean of risk, and they order studies differently. Members are
combined by rank, since §1 established order is all the metric reads.


## 7. Validating without fooling yourself

Two leaks are specific to this setup, and both inflate a validation number without
improving anything.

**Shared reports.** Some reports are byte-identical across studies — a template read for an
unremarkable knee — so every study in such a group receives the same derived target vector,
and splitting the group across the divide scores the model on a target whose source it was
trained on. Studies are therefore assigned by a hash of the report text, which keeps every
duplicate group whole. One fifth is held out, and the split is fixed rather than rotated.

**Two references, two meanings.** The **holdout** covers a fifth of the corpus, measures
agreement with the derived targets, and has enough studies per label to separate a real
difference from noise — it selects both the epoch within a run and the recipe between runs.
The **annotation check** measures agreement with a radiologist's reading of the images,
which is what the competition scores, but only the annotated studies falling in the holdout
can be used and there are very few; by the standard-error argument of §2 it is reported and
never allowed to arbitrate.

The annotated studies stay in training, at elevated weight, because they are the only labels
read from the images rather than from text. That is exactly why the annotation check must be
restricted to the holdout: scoring a model on training examples whose answers it saw,
weighted more heavily than anything else, measures memorisation and reports it as skill.


## 8. Independent Synovitis specialist

Version 26 adds a single evidence-qualified family, adapted with attribution from `blacklions/report-teacher-anatomy-aware-hierarchical-multimod`. Its frozen DINOv2-base RTA-HMIL ensemble uses six plane/sequence slots and seven 2.5D views per slot. On the 58 complete official annotation rows, its Synovitis AUC is 0.826 versus 0.742 for V25; repeated target-stratified four-fold selection chooses a positive weight in 99.95% of folds and gives a positive held-out delta in 99.4% of 500 partitions. The selected median weight is 0.75.

The specialist runs only after the complete V25 submission has been written. It atomically replaces only the Synovitis column after all eight checkpoints finish; missing inputs, decode failures, or a conservative 8.6-hour deadline leave the V25 file untouched. The other eleven targets are asserted bit-for-bit unchanged.

### V34: train-only PCA exact-label specialists

V34 preserves V33 and adds a separately validated, deploy-exact PCA arm. Scaling and PCA are fit on the 4,407 competition training studies only; four randomized PCA fits are averaged. Lateral Meniscus improved in every PCA-seed held block and the selected three-way blend improved V33 by 0.0273 AUC, with all leave-one-subject-out deltas positive. Synovitis improved by 0.0185 in the held block identically across all four PCA seeds. Fracture and Effusion were deliberately left unchanged because their evidence was less stable. These tests reuse the same 58 image-adjudicated subjects, so they measure estimator stability rather than independent clinical generalization.


## 9. What the scored `.903` EfficientNet-B3 reference changes

The strongest fully visible score above this notebook is Yash Bishnoi's
`yashbishnoi98/rsna-knee-infer-v1`, whose immutable Version 5 is attached to a completed
public score of **`.903`**.  The source now exposes enough of the experiment design to
separate the useful recipe from the part that cannot be reproduced.

### The score begins with better supervision

The model is trained on all 4,349 report-only studies rather than the 58 expert-labelled
studies alone.  Its primary report reader is a locally served Qwen3.6-35B model, run at
temperature zero with reasoning disabled and a fixed twelve-key JSON contract.  Prompt
revisions were measured on the 58 gold studies: the published final reader reports `0.833`
mean per-class accuracy and `0.766` mean F1.  Synovitis was broadened to include indirect
inflammatory language; trace effusion was treated more cautiously; degenerative marrow
oedema stopped counting as contusion without a traumatic pattern; and avulsion and
insufficiency fractures were made explicit.

Those labels are then fused, per target, with the public
`pilkwang/rsna-knee-llm-labels` reader.  Agreement stays hard; disagreement becomes a soft
target weighted by each reader's measured accuracy for that particular finding.  This is
the most important lesson from the `.903` system: the image model is downstream of label
quality, and one global reader weight throws away useful per-finding evidence.

### The image recipe that won its own ablation

Twenty proxy trials on 10% of the studies for two epochs ranked candidates by
`0.7 * CV AUC + 0.3 * gold58 AUC`.  The winner was not the most elaborate pooling option:

| choice | scored Version 5 recipe |
|---|---|
| backbone | single-channel ImageNet EfficientNet-B3 |
| folds | five study-grouped folds |
| study input | three fluid-sensitive, plane-diverse series |
| sampling | 12 slices/series in training; 32 at inference |
| pooling | max over slice embeddings, then mean over series logits |
| augmentation | shared rotation, gamma and scale; **no horizontal flip** |
| objective | BCE-with-logits on fused soft labels |
| optimisation | AdamW, short warm-up, cosine decay, bf16 |
| schedule | eight epochs per fold |
| ensemble | equal mean of five fold sigmoids |

The five-fold run took about 12.4 hours and reports clean macro OOF AUC `0.8544`.  The 58
gold studies were held out of training and cross-fitted so that each was predicted only by
a checkpoint that had not seen it; that independent macro AUC was `0.8568`.  Agreement
between those two disjoint checks is much stronger evidence than choosing architecture by
leaderboard feedback.

The scored V5 inference applies DICOM slope/intercept, inverts `MONOCHROME1`, sorts by
`InstanceNumber`, windows each series at its 1st/99th percentiles, center-crops depth at 64,
resizes to 288 pixels, and quantises through `uint8`.  It checkpoints a valid submission
every 25 studies and can degrade from 32 to 16 slices and then from five to three folds if
its projected runtime exceeds eight hours.  A later, unscored source revision adds
square-padding and fuller decode diagnostics; those changes must not be attributed to the
completed `.903` row.

### Evidence boundary and promotion contract

| artifact | what is public | what this notebook may claim |
|---|---|---|
| Pilkwang DINO family | complete inspectable code and 20 public checkpoints | independently executable `.891` anchor |
| this notebook | completed official rows | highest verified public score `.894` |
| public DINO output | exact attached predictions | independently completed `.899` reference, not yet this notebook's official score |
| Yash B3 family | exact inference plus training protocol and validation summaries | completed `.903` reference; not an exact reproduction |

The five B3 checkpoints, `model_code.py`, fused labels, and locked Qwen prompt remain in a
private input package.  The downloadable notebook output contains only the three visible
sample UIDs.  Therefore its hidden-test ordering cannot be imported or honestly claimed as
ours, and the runtime below accepts a Yash-family artifact only after exact UID-set, schema,
finite-value, error-count, and study-count checks.

An independent promotion must reproduce the supervision and not just the word
"EfficientNet": per-class label fusion, report-grouped folds, no laterality-breaking flips,
five full B3 folds, and disjoint OOF/gold validation.  Until such checkpoints exist and beat
the current macro OOF baseline, Version 40's audited public-frontier prediction remains the
safe primary output.


## 10. A guarded, independently trainable B3 branch

The `.903` reference is not a reusable output file: its decisive five checkpoints are private. This branch therefore trains an independent model from public, hash-pinned inputs. It encodes each study once with a frozen public EfficientNet-B3, trains five target-attention MIL heads on three-source soft-label consensus, and checks all 58 expert studies strictly out of fold. Physical DICOM ordering, a 130 mm crop, laterality normalization, three planes, and 32 slices are explicit contracts.

The branch is fail-safe. The independently reproduced public `.899` artifact is produced first and remains `submission.csv` if any input hash, training contract, 58-study OOF audit, or inference schema fails. Even a successful local OOF result is diagnostic evidence only; only a completed competition row can establish a public score.

## 11. Running B3 when Kaggle assigns an incompatible P100

The `.903` reference is not a reusable output file: its decisive five checkpoints are private. This retry keeps the same independent public inputs but probes actual CUDA execution and uses a bounded 192px, four-slice-per-plane CPU cache when the assigned GPU is incompatible. It encodes each study once with a frozen public EfficientNet-B3, trains five target-attention MIL heads on three-source soft-label consensus, and checks all 58 expert studies strictly out of fold. Physical DICOM ordering, a 130 mm crop, laterality normalization, three planes, and 32 slices are explicit contracts.

The branch is fail-safe. The independently reproduced public `.899` artifact is produced first and remains `submission.csv` if any input hash, training contract, 58-study OOF audit, or inference schema fails. Even a successful local OOF result is diagnostic evidence only; only a completed competition row can establish a public score.

## 12. Two-T4 final-block fine-tuning from the public `.903` recipe

The scored `.903` notebook discloses its full pipeline but not its five private fold weights. This independent branch retains its decisive public choices: EfficientNet-B3, one fluid-sensitive series per plane, InstanceNumber ordering, 1st–99th percentile windowing, square padding, uint8 roundtrip, five group-aware folds, no horizontal flip, and max-sensitive multi-instance pooling. The official versioned timm backbone is hash-pinned, and folds are trained in waves on two T4s.

All 58 expert studies are scored strictly out of fold. The established `.899` public-specialist artifact remains `submission.csv`; standalone and rank-blended V47 candidates are separate outputs until an official completed competition row can establish their score.

## 13. Minimum rank crossing for two independently supported findings

Version 52 established an official public score of **0.906** with a 20%
percentile-rank vote from the independent RadImageNet ResNet-50 family. A
blanket increase is not justified: on the visible test identities, 42.5% would
change only Effusion and Fracture, while both lose OOF AUC at that weight.

The fold-isolated audit supports a narrower change. At the smallest robust
weight above the 50% rank crossover, ACL improves in three held folds and harms
none; Synovitis improves in four of five. All other findings remain at the
already verified 20%. The cell below uses **50.5%** for only those two heads,
checks the pinned 58-study OOF evidence again on Kaggle, and verifies that only
their test rank orders change. Any failed contract restores the exact 20%
artifact. These are diagnostic labels only; a completed competition row is the
sole leaderboard evidence.

When Kaggle assigns CPU and the earlier GPU intermediates are absent, the implementation replays the exact hash-pinned three-study V62 artifact produced by the same audited inference stack; schema, study order, and content hash must all match before promotion.


## 14. Frozen-head external-encoder frontier

The next cell performs inference only. It loads hash-pinned fold heads from the completed BiomedCLIP and fastMRI+ ViTDet experiments, preserves the V62 RadImageNet rank-crossing parent, and changes only four findings supported by non-negative diagnostics under both fold partitions. The 58-label audit is explicitly descriptive rather than an unbiased leaderboard estimate.


## TPU frontier: Curia-2, guarded by the exact 0.906 anchor

Version 52 remains this notebook's best completed official result at **0.906**. The
label-wise multi-encoder overlay subsequently scored **0.887**, falsifying broad
post-selection on only 58 image-labelled studies.

This experiment uses the official Curia-2 radiology encoder across each MRI series,
five report-grouped folds, and one nested global blend weight. The account's weekly GPU
quota is exhausted, so the frozen PyTorch encoder and small query heads run on Kaggle's
unused TPU quota through PyTorch/XLA. Before XLA is initialised, the code restores the
exact audited Version 52 submission; every missing-input, accelerator, runtime, OOF,
time-budget, or schema failure restores that same byte-identical anchor.

Grouped-fold diagnostics are local selection evidence, not a public score. A completed
competition submission with a non-empty score is the only evidence of improvement.
