# H-B-182 L1 RCO full-cache v1.1.0

Changelog: CPU-only parallel full-roster cache and deterministic six-study proof;
pre-release review fixes bind proof SHA256 and lock serial-reference execution.
The three v1.0.0 tools and Stage 0 card remain byte-identical; no originals or
scored stages are edited. No full-cache execution is authorized by this card
until B has a passing K0 and independent result/admission review.

## Contract and limits

The registered cohort is exactly 3190 train + 977 holdout = 4167 studies.
The source registration, component/Gold exclusions, full train/holdout UID hashes,
renderer/px_cache/anatomy/anchor/init and raw archive entry identities are sealed.
The raw train_series.csv metadata is hashed as well. Production register/build
never open or hash label files; extracted train.csv contains only the study UID.
The untouched v1 serial proof retains its legacy target-file byte-hash validation,
without parsing target values; this exception is confined to that reference.
EXPOSED_EXPLORATORY;
clean_oof=false; anchor ancestry remains unresolved.

Default execution uses six spawned Windows worker processes, each with two torch
intra-op threads and one inter-op thread. Six workers are a ceiling; B may use
--workers 3 or --workers 4 when GPU jobs' decoding competes for CPU/RAM/IO.
Available RAM must be >=2 GiB per worker +4 GiB. This is an admission margin,
not a measured upper bound on process RAM. Default threads occupy 12 of the
machine's 16 logical threads; eight physical cores do not imply sixfold scaling.

Each worker has one fixed scratch slot under .local/scratch/l1-rco/fullcache/workers.
One nonblocking kernel lock prevents overlapping cache invocations and scratch
reuse. Up to N studies are outstanding. Each shard writes atomic NPY payloads,
then its checksum receipt as the commit marker. An interrupted uncommitted study
is rebuilt; committed shards must pass hashes/layout/UID/contract checks and are
not rewritten. Corrupt committed shards cause a hard failure, never silent reuse.

Output is fixed to G:/temp_data/rsna-knee/l1rco-cache-full. Proof outputs stay
inside the fixed in-project task folder. Rendering reserves 4 GiB output plus
4 GiB extraction margin per concurrent worker (combined on the same volume).
Assembly separately requires the complete mmap payload plus 2 GiB; retained
shards make steady disk use approximately twice the final cache, about 740 GB
for a 370 GB payload. Partial assembly restarts using verified shards. Final
receipt is committed after all mmap/index hashes. Progress/status and ETA are
reported every <=15 seconds during rendering; ETA is based on new completions,
includes startup, and does not predict the subsequent assembly/hash phase.
Shard verification and assembly copying also log progress/phase ETA at study
boundaries (at least every 15 seconds when a study operation has completed).
Final whole-file checksum time has no ETA. Disk gates reserve each concurrent
study and assembly, not the full cohort in advance; disk exhaustion fails safely.

The assembly uses roster order, identical uint8 pixels, float32 Nx12 metadata,
and original global view offsets/index schema. Contract IDs intentionally differ
from Stage 0, so a future full-fit trainer must be separately prepared/reviewed;
this does not alter or authorize the 200-step K0 trainer or any full fit.

## CPU commands and six-study proof

Run in project root. All entrypoints set CUDA_VISIBLE_DEVICES empty before torch.

```powershell
$l1py = 'C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe'
& $l1py -m pytest tools/test_l1_rco_fullcache_v110.py tools/test_l1_rco_v1.py tests/test_exp189_s6_port.py tests/test_gpu_queue.py tests/test_s6plus.py --basetemp=.local/pytest-tmp/l1-rco -q
& $l1py tools/l1_rco_fullcache_v110.py register
& $l1py tools/l1_rco_fullcache_v110.py validate
& $l1py tools/l1_rco_fullcache_v110.py proof
& $l1py tools/l1_rco_fullcache_v110.py admit
```

The last command deliberately fails closed with no K0/result-review arguments.
There is no prepare command or GPU launch path in this family. The proof selects
four train + two holdout studies. It invokes the untouched v1 serial renderer in
a fresh subprocess, rebinding only its WORK containment path. It compares every
study's px.npy/meta.npy bytes to the six-worker path, plus final mmap bytes and
index contents (except the intentional contract ID). A second six-study output
stops after three committed studies, resumes and verifies unchanged committed
receipts and identical final payloads. Synthetic tests additionally corrupt
shards and interrupt assembly. No full cohort image build runs in these checks.

## Independent review and B-only full execution

B must reconcile CPU/IO/RAM and current compute ownership before execution.
After K0, an independent reviewer writes the actual evidence below to
.local/scratch/l1-rco/fullcache/FULLCACHE-REVIEW.json. These placeholders do not
constitute admission. This review covers both K0 result and exact full-cache
implementation/proof, not merely the Stage 0 implementation.

```json
{
  "verdict": "PASS",
  "builder": "l1-rco",
  "reviewer": "independent-reviewer",
  "contract_id": "fullcache registration contract_id",
  "K0_result_sha256": "actual SHA256 of K0 result.json",
  "proof_sha256": "actual SHA256 of fullcache/proof.json",
  "report": ".local/scratch/l1-review/fullcache/ADMISSION-REVIEW.md",
  "report_sha256": "actual report SHA256"
}
```

K0 must satisfy the original exact 200-step, finite/decreasing loss, >=80windows/s,
peak <=15 GB AND <=12 GiB and baseline+peak+2GiB<=total rules. Source registration,
numeric rule, result hash, full-cache contract and independent reviewer identity
are checked before full execution. No fabricated PASS is supplied with this build.

```powershell
& $l1py tools/l1_rco_fullcache_v110.py admit --k0-result .local/scratch/l1-rco/stage0/k0/result.json --review .local/scratch/l1-rco/fullcache/FULLCACHE-REVIEW.json
& $l1py tools/l1_rco_fullcache_v110.py build --workers 6 --out G:/temp_data/rsna-knee/l1rco-cache-full --k0-result .local/scratch/l1-rco/stage0/k0/result.json --review .local/scratch/l1-rco/fullcache/FULLCACHE-REVIEW.json
```

Repeat the same build command after interruption; committed shards resume after
checksum validation. Full-cache size/time and live workload interference remain
unmeasured until B's authorized run. From the supplied 13.48s/study serial rate,
4167*13.48/3600 =15.60h serial and ideal six-way rendering is 2.60h, plus startup,
assembly/checksums and contention. This arithmetic is not a measured wall-time
promise. Final six-study parallel execution took 40.2533s including startup,
assembly and hashes: naive linear extrapolation is 7.77h, which overcounts
repeated startup and does not establish sustained throughput or an upper bound.
Use approximately 7.8h as a provisional scheduling estimate, then update from
actual full-run ETA. Detailed evidence is in
`.local/scratch/l1-rco/fullcache/REPORT.md`; full execution is UNMEASURED.
No Kaggle T4 serving or full-fold fit time follows from this CPU measurement.
