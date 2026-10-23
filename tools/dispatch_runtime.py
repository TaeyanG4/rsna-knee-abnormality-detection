"""Persistent Codex lane protocol, v1.3.1 (2026-10-13).

Changelog: atomic reservation, Bash-compatible lock PID, immutable run folders,
successful-completion counts and terminal receipt waiter. No session-file reads.
v1.1.0: explicit audited release after Windows kernel process-tree confirmation.
v1.2.0: force UTF-8 console output; expose reserved lanes before session creation.
v1.2.1 (B): ACCESS RULES pin pytest --basetemp to one fixed C: folder per lane (pytest wipes it each run); unique H: basetemps filled H: on 10-12.
v1.2.2 (B): large scratch/verification outputs also go to one fixed C:/kaggle-offload/<lane>/<task> folder.
v1.2.3 (B): pytest basetemp back on H: as ONE fixed .local/pytest-tmp/<lane> folder (some tests need project-relative tmp paths; the disk problem was unique per-run folders, not the drive).
v1.2.4 (B, Taeyang): nothing leaves the project folder on H:; large scratch goes to one fixed .local/scratch/<lane>/<task> folder.
v1.3.0: astra builder aliases, explicit idle-only migration, bounded fresh overflow sessions.
v1.3.1: reject direct astra-xN dispatch; overflow allocation is internal on BUSY only.
DISPATCH_MAX_OVERFLOW is configurable (default 6), not a hard ceiling of six.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import job_runtime as rt

VERSION = '1.3.1'
ALIASES = {f'sol-{i}': f'astra-b{i}' for i in range(1, 6)}
LANES = (*ALIASES.values(), 'astra-1','astra-2','astra-3','astra-4','astra-5','astra-6','extra')
LANE_SUFFIXES = ('id','last','model','n','opened','state.json','lock')
ROLES = {'astra-b1':'RELEASE implementer (finals, pre-reading checks, B174, submission tooling)',
         'astra-b2':'DATA/TRAINING implementer (OAI/external caches, experiments, analysis tools)',
         'astra-b3':'SCOUT (public notebooks, forum, information)',
         'astra-1':'VERIFIER for astra-b1 (release/finals)', 'astra-2':'VERIFIER for astra-b2 (data/training)',
         'astra-3':'REVIEWER (cross-checks, decision consults, audits)'}
ACCESS = '''

ACCESS RULES (B 2026-10-11 v2.0.0): you have FULL access — read, write and execute anything needed, including the GPU Python C:/Users/Taeyang/AppData/Local/Programs/Python/Python312/python.exe, project tests and CPU dry-runs; run your own tests before reporting. Original/frozen project files MAY be edited when needed (bump the version, add a changelog line, keep a backup copy under .local/backups/, say so in the report); never edit already scored release stages or receipts in place (make a new versioned stage). Verifier/reviewer jobs: execute freely but change no project file except your report. HARD LIMITS: never read or print credential files (H:/kaggle/api-keys, ~/.kaggle, ~/.codex, ~/.ssh keys, Vast keys); never open radiology report text or .local/relabel, never print label rows; never kaggle push / submit / dataset upload and never launch GPU training (B does); never delete user data or raw/competition data; always run pytest with --basetemp=.local/pytest-tmp/<your lane name> (exactly this one fixed folder per lane; pytest clears it at each run; never a new unique basetemp); put any other large scratch or real-data verification output (copied stages, smoke runs) in one fixed folder .local/scratch/<your lane name>/<task> inside the project that you overwrite on re-runs — never new unique folders and never any location outside the project folder (no C: or other drives); never recreate, sync or reinstall the project .venv (uv run/uv sync/uv venv/python -m venv) — ask B for package changes. BUILDERS: before reporting, run the full relevant test suite, a real-data CPU smoke of every new code path, and every CPU command of your LAUNCH card (register/validate/prepare/admit dry-runs) yourself; paste their results. VERIFIERS: review the DIFF and the specific questions asked; do not re-run what the pasted builder results already prove unless you doubt them; keep the report short.
'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stamp():
    return dt.datetime.now().strftime('%m-%d %H:%M')


def canonical_lane(lane, *, notice=False):
    if notice and lane in ALIASES:
        print(f'DEPRECATED lane {lane}; use {ALIASES[lane]}', file=sys.stderr)
    lane = ALIASES.get(lane, lane)
    if lane not in LANES and not re.fullmatch(r'astra-x[1-9][0-9]*', lane):
        raise ValueError('lane must be one of: '+' '.join(LANES)+' or astra-xN (sol-1..sol-5 aliases accepted)')
    return lane


def is_overflow(lane):
    return bool(re.fullmatch(r'astra-x[1-9][0-9]*', lane))


def overflow_limit():
    value = os.environ.get('DISPATCH_MAX_OVERFLOW', '6')
    if not value.isdecimal():
        raise ValueError('DISPATCH_MAX_OVERFLOW must be a nonnegative integer')
    return int(value)


def overflow_metadata(job):
    if not is_overflow(job['lane']):
        return None
    path = Path(job['run'])/'overflow.json'
    if job['sources'].get(str(path)) != sha(path):
        raise ValueError('overflow assignment evidence differs')
    metadata = rt.read_json(path)
    if (metadata.get('schema') != 'dispatch-overflow/v1' or
            type(metadata.get('keep_session')) is not bool):
        raise ValueError('invalid overflow assignment')
    canonical_lane(metadata['requested_lane'])
    return metadata


def bash():
    configured = os.environ.get('KNEE_BASH')
    if os.name == 'nt' and (not configured or configured.startswith('/')):
        return 'C:/Program Files/Git/bin/bash.exe'
    return configured or 'bash'


def launch_options():
    return {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}


def publication_paths(out):
    out = Path(out)
    log = out.with_name(out.name[:-3]+'.log' if out.name.endswith('.md') else out.name+'.log')
    return [out, log, *(Path(str(out)+suffix) for suffix in ('.prompt','.receipt.json','.dispatch.json'))]


def windows_jobs():
    """Named jobs retain descendants even when intermediate parents are killed."""
    import ctypes as c
    from ctypes import wintypes as w
    kernel = c.WinDLL('kernel32', use_last_error=True)
    for name, args, result in (
        ('CreateJobObjectW', [w.LPVOID, w.LPCWSTR], w.HANDLE),
        ('OpenJobObjectW', [w.DWORD, w.BOOL, w.LPCWSTR], w.HANDLE),
        ('AssignProcessToJobObject', [w.HANDLE, w.HANDLE], w.BOOL),
        ('QueryInformationJobObject', [w.HANDLE, c.c_int, w.LPVOID, w.DWORD, w.LPVOID], w.BOOL),
        ('GetCurrentProcess', [], w.HANDLE), ('CloseHandle', [w.HANDLE], w.BOOL),
    ):
        function = getattr(kernel, name)
        function.argtypes, function.restype = args, result
    return c, kernel


def tree_guardian(run):
    """Keep the named object observable if the worker and parents are killed."""
    run = Path(run)
    job = rt.read_json(run/'job.json')
    c, kernel = windows_jobs()
    name = 'Local\\knee-dispatch-' + job['run_id']
    handle = kernel.CreateJobObjectW(None, name)
    if not handle: raise c.WinError(c.get_last_error())
    if c.get_last_error() == 183:
        kernel.CloseHandle(handle)
        raise ValueError('process-tree object already exists')
    guardian = dict(schema='dispatch-tree-guardian/v1', run_id=job['run_id'],
                    name=name, identity=rt.identity())
    rt.atomic_json(run/'tree-guardian.json',guardian)
    try:
        while True:
            job=rt.read_json(run/'job.json')
            if job.get('containment'):
                state=contained_tree_state(job['containment'],job['run_id'])
                if state=='dead':
                    rt.atomic_json(run/'tree-terminal.json',dict(schema='dispatch-tree-terminal/v1',
                        run_id=job['run_id'],containment=job['containment'],worker=job['worker'],
                        guardian=guardian,state='dead',completed=dt.datetime.now().isoformat()))
                    return 0
                if state=='unknown': raise ValueError('cannot observe process tree')
            elif rt.liveness(job['owner'])=='dead' and job['phase']=='reserved':
                # Spawn/handshake failure has no containment proof; keep held.
                return 1
            time.sleep(.1)
    finally:
        kernel.CloseHandle(handle)


def contain_worker(run):
    # No kill-on-close: release never terminates processes. The kernel object
    # persists while any member survives, including orphaned grandchildren.
    if os.name != 'nt':
        return None
    guardian=rt.read_json(Path(run)/'tree-guardian.json')
    if rt.liveness(guardian['identity'])!='alive':
        raise ValueError('process-tree guardian is unavailable')
    c, kernel = windows_jobs()
    name = guardian['name']
    handle = kernel.OpenJobObjectW(5,False,name)  # ASSIGN_PROCESS | QUERY
    if not handle:
        raise c.WinError(c.get_last_error())
    if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        error = c.get_last_error()
        kernel.CloseHandle(handle)
        raise OSError(error, 'cannot establish unique process-tree containment')
    # Keep the handle open until worker exit. All later child processes inherit
    # this job; no breakaway permission is enabled.
    return dict(kind='windows-job/v1', name=name)


def contained_tree_state(containment, run_id):
    if (os.name != 'nt' or not isinstance(containment, dict) or
        containment != dict(kind='windows-job/v1', name='Local\\knee-dispatch-'+run_id)):
        return 'unknown'
    c, kernel = windows_jobs()
    handle = kernel.OpenJobObjectW(4, False, containment['name'])  # QUERY access
    if not handle:
        return 'unknown'
    from ctypes import wintypes as w
    class Accounting(c.Structure):
        _fields_ = [(name, c.c_int64) for name in ('user','kernel','period_user','period_kernel')] + [
            (name, w.DWORD) for name in ('faults','total','active','terminated')]
    info = Accounting()
    try:
        if not kernel.QueryInformationJobObject(handle, 1, c.byref(info), c.sizeof(info), None):
            return 'unknown'
        return 'dead' if info.active == 0 else 'alive'
    finally:
        kernel.CloseHandle(handle)


class Lanes:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root/'.local/astra-improve/lanes'
        self.directory.mkdir(parents=True, exist_ok=True)

    def raw_file(self, lane, suffix):
        return self.directory/(lane+'.'+suffix)

    def migration_file(self, lane):
        return self.directory/(lane+'.migration.json')

    def storage_lane(self, lane):
        lane = canonical_lane(lane)
        if self.migration_file(lane).exists():
            raise ValueError(f'lane {lane} migration interrupted; run dispatch.sh migrate')
        old = next((old for old, new in ALIASES.items() if new == lane), None)
        if old and any(self.raw_file(old, s).exists() for s in LANE_SUFFIXES):
            if any(self.raw_file(lane, s).exists() for s in LANE_SUFFIXES):
                raise ValueError(f'lane {lane} has conflicting legacy and canonical records')
            return old
        return lane

    def file(self, lane, suffix):
        return self.raw_file(self.storage_lane(lane), suffix)

    def inventory(self):
        # Include retained overflow records even if the configured cap is lowered.
        extra = {p.name.split('.')[0] for p in self.directory.glob('astra-x*.*')
                 if is_overflow(p.name.split('.')[0])}
        return (*LANES, *sorted(extra, key=lambda s: int(s[7:])))

    def read(self, lane, suffix, default=''):
        file = self.file(lane, suffix)
        return file.read_text(encoding='utf-8').strip() if file.exists() else default

    def role(self, lane):
        lane = canonical_lane(lane)
        if is_overflow(lane):
            state = self.file(lane, 'state.json')
            metadata = overflow_metadata(rt.read_json(state)) if state.exists() else None
            return 'OVERFLOW for '+(metadata['requested_lane'] if metadata else 'unassigned')
        if lane == 'extra':
            return f'EXTRA 7th session ({self.read(lane,"model","unset")}; temporary — close when its item is done)'
        return ROLES.get(lane, 'POOL implementer (overflow builds/fixes)' if lane.startswith('astra-b') else 'POOL verifier (one item per job)')

    def mutex(self):
        return rt.file_lock(self.directory/'.registry-mutex')

    def busy(self, lane):
        return self.busy_storage(self.storage_lane(lane))

    def busy_storage(self, lane):
        state_path = self.raw_file(lane, 'state.json')
        if state_path.exists():
            state = rt.read_json(state_path)
            if state.get('phase') != 'terminal':
                # Unknown/dead owner in the handoff window cannot prove that a
                # worker was never spawned. Fail closed; never reclaim by age.
                return True
        lock = self.raw_file(lane, 'lock')
        pid = lock.read_text(encoding='utf-8').strip() if lock.exists() else ''
        if pid:
            if not pid.isdecimal():
                return True
            # Existing .lock files contain MSYS PIDs, not native Windows PIDs.
            return subprocess.run([bash(), '-c', 'kill -0 "$1" 2>/dev/null', 'lane-check', pid],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **launch_options()).returncode == 0
        return False

    def migrate(self):
        # The existing mutex serializes all cooperating writers. A durable journal
        # blocks alias resolution during partial moves; rerunning completes them.
        # No job/run/receipt contents change and BUSY namespaces are never moved.
        with self.mutex():
            for old, new in ALIASES.items():
                journal = self.migration_file(new)
                if self.busy_storage(old) or self.busy_storage(new):
                    print(f'{old} -> {new}: BUSY; deferred (files unchanged)')
                    continue
                if journal.exists():
                    record = rt.read_json(journal)
                    if (record.get('schema') != 'dispatch-migration/v1' or
                            record.get('old') != old or record.get('new') != new or
                            not set(record.get('files', {})) <= set(LANE_SUFFIXES)):
                        raise ValueError(f'invalid migration journal for {new}')
                    files = record['files']
                else:
                    files = {s: sha(self.raw_file(old, s)) for s in LANE_SUFFIXES
                             if self.raw_file(old, s).exists()}
                    if not files:
                        print(f'{old} -> {new}: already migrated or empty')
                        continue
                    if any(self.raw_file(new, s).exists() for s in LANE_SUFFIXES):
                        raise ValueError(f'migration target {new} already has records; unchanged')
                    rt.atomic_json(journal, dict(schema='dispatch-migration/v1', old=old, new=new, files=files))
                # Validate the entire transaction before moving any remaining file.
                for suffix, digest in files.items():
                    source, target = self.raw_file(old, suffix), self.raw_file(new, suffix)
                    if source.exists() == target.exists() or sha(source if source.exists() else target) != digest:
                        raise ValueError(f'migration evidence differs for {new}.{suffix}')
                for suffix in files:
                    source, target = self.raw_file(old, suffix), self.raw_file(new, suffix)
                    if source.exists():
                        rt.replace_file(source, target)
                journal.unlink()  # Commit: aliases now resolve entirely to new files.
                print(f'{old} -> {new}: migrated {len(files)} files')
            self.registry()
        return 0

    def registry(self):
        lines = ['# Codex session registry (generated by dispatch.sh v3.2.0; closed sessions are removed)', '',
                 '| Lane | Role | Session | Opened | Jobs | Last job | State |',
                 '|---|---|---|---|---|---|---|']
        for lane in self.inventory():
            session = self.read(lane, 'id')
            if session or self.busy(lane):
                state = 'BUSY pid '+self.read(lane,'lock','reserved') if self.busy(lane) else 'idle'
                lines.append(f'| {lane} | {self.role(lane)} | {session} | {self.read(lane,"opened")} | {self.read(lane,"n","0")} | {self.read(lane,"last")} | {state} |')
        rt.atomic_text(self.directory/'REGISTRY.md', '\n'.join(lines)+'\n')

    def clear_session(self, lane):
        # These are dispatcher metadata, not reports or user data. Preserve a
        # copy for each rotation/reset before removing the active record.
        records = {s:self.read(lane,s) for s in ('id','n','opened','last','model') if self.file(lane,s).exists()}
        if records:
            rt.atomic_json(self.directory/'history'/f'{lane}-{uuid.uuid4().hex}.json', records)
        for suffix in records:
            self.file(lane,suffix).unlink()

    def close(self, lane):
        lane = canonical_lane(lane, notice=True)
        with self.mutex():
            if self.busy(lane):
                print(f'lane {lane} is BUSY; not closed'); return 3
            self.clear_session(lane); self.registry()
        print(f'lane {lane} closed (record deleted)'); return 0

    def release(self, lane):
        """Administrative failure only, never success, retry, or process killing."""
        lane = canonical_lane(lane, notice=True)
        with self.mutex():
            state_file = self.file(lane, 'state.json')
            if not state_file.exists():
                print(f'lane {lane} NOT_RELEASED: no process-tree provenance'); return 3
            job = rt.read_json(state_file)
            if job.get('phase') == 'terminal':
                print(f'lane {lane} already terminal; no change'); return 0
            run_id = job.get('run_id', '')
            if not re.fullmatch('[0-9a-f]{32}', run_id):
                raise ValueError('invalid reservation identity')
            if canonical_lane(job['lane']) != lane:
                raise ValueError('reservation lane differs')
            run = self.directory/'runs'/job['lane']/run_id
            if job.get('run') != str(run) or rt.read_json(run/'job.json') != job:
                raise ValueError('reservation evidence differs')
            processes = {key:rt.liveness(job.get(key)) for key in ('owner','worker')}
            if job.get('child') is not None:
                processes['child'] = rt.liveness(job['child'])
            tree = contained_tree_state(job.get('containment'), run_id)
            terminal=run/'tree-terminal.json'
            if tree=='unknown' and terminal.exists():
                proof=rt.read_json(terminal)
                guardian=rt.read_json(run/'tree-guardian.json')
                if (proof.get('schema')=='dispatch-tree-terminal/v1' and proof.get('run_id')==run_id and
                    proof.get('containment')==job.get('containment') and proof.get('worker')==job.get('worker') and
                    proof.get('guardian')==guardian and proof.get('state')=='dead'):
                    tree='dead'
            lock = self.read(lane, 'lock')
            bash_dead = (not lock or (lock.isdecimal() and lock == str(job.get('bash_pid')) and
                subprocess.run([bash(), '-c', 'kill -0 "$1" 2>/dev/null', 'lane-check', lock],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               **launch_options()).returncode == 1))
            if tree != 'dead' or not bash_dead or any(v != 'dead' for v in processes.values()):
                print(f'lane {lane} NOT_RELEASED: process tree alive or unconfirmed'); return 3
            evidence = dict(schema='dispatch-release/v1', run_id=run_id, lane=job['lane'],
                action='release_after_process_tree_death', completed=dt.datetime.now().isoformat(),
                actor=rt.identity(), previous_job=job, processes=processes, tree=tree,
                bash_dead=bash_dead, success=False, completed_count=self.read(lane,'n','0'))
            # Preserve any partial worker receipt and every report/log unchanged.
            if (run/'release.json').exists():
                if rt.read_json(run/'release.json')['previous_job']!=job:
                    raise ValueError('prior release evidence differs')
            else:
                rt.atomic_json(run/'release.json', evidence)
            job = dict(job, phase='terminal', release=dict(path=str(run/'release.json'),
                                                         sha256=sha(run/'release.json')))
            rt.atomic_json(run/'job.json', job)
            rt.atomic_json(state_file, job)
            self.file(lane, 'lock').unlink(missing_ok=True)
            metadata = overflow_metadata(job)
            if metadata and not metadata['keep_session']:
                self.clear_session(lane)
            self.registry()
        print(f'lane {lane} released (process tree dead; failure recorded)'); return 0

    def dispatch(self, args):
        lane = args[0]
        if lane == 'migrate':
            if len(args) != 1: raise ValueError('usage: migrate')
            return self.migrate()
        if lane == 'lanes':
            with self.mutex():
                self.registry()
                print((self.directory/'REGISTRY.md').read_text(encoding='utf-8'), end='')
                for item in self.inventory():
                    if not self.read(item,'id') and not self.busy(item):
                        print(f'{item}: no open session — {self.role(item)}')
            return 0
        if lane in ('close','reset'):
            return self.close(args[1] if len(args)>1 else '')
        if lane == 'release':
            if len(args) != 2: raise ValueError('usage: release <lane>')
            return self.release(args[1])
        if is_overflow(lane):
            raise ValueError(f'cannot dispatch directly to {lane}: overflow lanes are allocated internally only when a requested lane is BUSY; dispatch to a regular lane instead')
        if len(args)<3:
            raise ValueError('missing prompt or out-file')
        prompt, out = Path(args[1]), Path(args[2])
        prompt = prompt if prompt.is_absolute() else self.root/prompt
        out = out if out.is_absolute() else self.root/out
        prompt, out = prompt.resolve(), out.resolve()
        who = args[3] if len(args)>3 else ''
        extra_model = who if lane == 'extra' else ''
        if lane == 'extra':
            if extra_model not in ('sol','astra'):
                raise ValueError('extra needs 4th arg sol|astra')
            who = args[4] if len(args)>4 else ''
        if lane == 'workspace-write': lane='astra-b1'
        if lane == 'read-only': lane='astra-b3' if who == 'sol' else 'astra-1'
        lane = canonical_lane(lane, notice=True)
        if not prompt.is_file(): raise ValueError('missing prompt '+str(prompt))
        model = 'gpt-6-sol' if lane == 'extra' and extra_model == 'sol' else 'gpt-6-astra'
        if (lane.startswith('astra-') and not lane.startswith('astra-b') or lane=='extra') and (self.directory/'NO-ASTRA').exists(): model='gpt-6-sol'
        notes=[]
        if who=='nosubagents': notes.append('NOTE: sub-agents hard-disabled for this job')
        if os.environ.get('FORCE_MODEL'):
            model=os.environ['FORCE_MODEL']; notes.append(f'NOTE: model forced to {model} for this job')
        if os.environ.get('SIMPLE')=='1':
            model='gpt-6-sol'; notes.append('NOTE: SIMPLE=1 -> gpt-6-sol for this simple job')
        with self.mutex():
            requested_lane = lane
            if self.busy(lane):
                if lane == 'extra' or os.environ.get('DISPATCH_NO_OVERFLOW') == '1':
                    print(f'BUSY lane {lane} (pid {self.read(lane,"lock","reserved")}); no new session started'); return 3
                limit = overflow_limit()
                active = sum(self.busy(item) for item in self.inventory() if is_overflow(item))
                candidates = (f'astra-x{i}' for i in range(1, limit+1))
                lane = next((item for item in candidates if not self.busy(item)), None) if active < limit else None
                if lane is None:
                    print(f'BUSY lane {requested_lane}; overflow cap reached; no new session started'); return 3
                notes.append(f'OVERFLOW: requested {requested_lane}; actual lane {lane}; out-file unchanged')
            if is_overflow(lane) and int(lane[7:]) > overflow_limit():
                raise ValueError('overflow lane exceeds DISPATCH_MAX_OVERFLOW')
            if is_overflow(lane) and sum(self.busy(item) for item in self.inventory()
                                        if is_overflow(item)) >= overflow_limit():
                print(f'BUSY lane {requested_lane}; overflow cap reached; no new session started'); return 3
            if lane=='extra' and self.read(lane,'id') and self.read(lane,'model') != extra_model:
                raise ValueError(f'extra lane is open as {self.read(lane,"model")}; close it first')
            session = self.read(lane,'id')
            count = int(self.read(lane,'n','0'))
            if session and count>=3:
                self.clear_session(lane); session=''; count=0
                print(f'lane {lane} closed after 3 jobs; opening a fresh session')
            previous_model = self.read(lane,'model','gpt-6-astra' if lane.startswith('astra-') else 'gpt-6-sol')
            if lane!='extra' and session and previous_model!=model:
                self.clear_session(lane); session=''; count=0
                print(f'lane {lane} was open on another model; opening a fresh {model} session')
            # Include derived log/sidecar names (foo and foo.md share foo.log).
            requested = {os.path.normcase(str(p)) for p in publication_paths(out)}
            for other in self.inventory():
                file = self.file(other,'state.json')
                if file.exists():
                    prior=rt.read_json(file)
                    occupied = {os.path.normcase(str(p)) for p in publication_paths(prior['out'])} if prior.get('out') else set()
                    if prior.get('phase')!='terminal' and requested & occupied:
                        print(f'BUSY lane {other} (pid {self.read(other,"lock","reserved")}); no new session started'); return 3
            if is_overflow(lane):
                self.clear_session(lane)
                session = ''; count = 0  # Overflow always starts a fresh session.
            run_id=uuid.uuid4().hex
            run=self.directory/'runs'/lane/run_id; run.mkdir(parents=True)
            log=publication_paths(out)[1]
            # Old reports/logs/prompts/receipts survive every output-name reuse.
            for source in publication_paths(out):
                if source.exists():
                    archive=run/'previous'/source.name; archive.parent.mkdir(exist_ok=True)
                    shutil.copyfile(source,archive)
            text=prompt.read_text(encoding='utf-8-sig')+ACCESS
            rt.atomic_text(run/'prompt.txt',text)
            rt.atomic_text(str(out)+'.prompt',text)
            state=dict(schema='dispatch-job/v1',run_id=run_id,lane=lane,phase='reserved',out=str(out),log=str(log),
                       run=str(run),session=session,model=model,count=count,owner=rt.identity(),worker=None,
                       child=None,bash_pid=None,prompt_sha256=sha(run/'prompt.txt'),
                       sources={str(Path(__file__).resolve()):sha(__file__),str(Path(rt.__file__).resolve()):sha(rt.__file__)},
                        effort='xhigh' if os.environ.get('EFFORT')=='xhigh' else 'high',nosubagents=who=='nosubagents')
            if is_overflow(lane):
                metadata = run/'overflow.json'
                rt.atomic_json(metadata, dict(schema='dispatch-overflow/v1', requested_lane=requested_lane,
                    keep_session=os.environ.get('DISPATCH_KEEP_OVERFLOW') == '1'))
                state['sources'][str(metadata)] = sha(metadata)
            for shim in ('dispatch.sh','wait-out.sh'):
                source=rt.CODE_ROOT/'.local/astra-improve'/shim
                state['sources'][str(source)] = sha(source)
            rt.atomic_json(run/'job.json',state)
            rt.atomic_json(self.file(lane,'state.json'),state)
            rt.atomic_json(str(out)+'.dispatch.json',dict(schema='dispatch-pointer/v1',run_id=run_id,lane=lane,run=str(run)))
            # Reservation is visible before process creation. Worker exclusively
            # publishes .lock using its own Bash PID; parent never overwrites it.
            rt.atomic_text(self.file(lane,'model'),extra_model if lane=='extra' else model)
            rt.atomic_text(self.file(lane,'last'),stamp()+' '+out.name+'\n')
            self.registry()
        command=[bash(),'-c','"$1" -B "$2" _worker "$3" "$4" "$BASHPID"; exit $?',
                 'lane-worker',sys.executable,str(Path(__file__).resolve()),str(self.root),str(run)]
        try:
            if os.name=='nt':
                with (run/'tree-guardian.log').open('ab') as guardian_log:
                    guardian=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'_guardian',str(run)],
                        cwd=self.root,stdin=subprocess.DEVNULL,stdout=guardian_log,stderr=subprocess.STDOUT,**launch_options())
                while not (run/'tree-guardian.json').exists():
                    if guardian.poll() is not None: raise ValueError('process-tree guardian failed; lane held')
                    time.sleep(.05)
            with (run/'worker.log').open('ab') as worker_log:
                child=subprocess.Popen(command,cwd=self.root,stdin=subprocess.DEVNULL,stdout=worker_log,
                                       stderr=subprocess.STDOUT,**launch_options())
        except OSError:
            # Reservation intentionally retained: no ambiguous automatic retry.
            raise
        while True:
            current=rt.read_json(run/'job.json')
            if current.get('bash_pid'): break
            if child.poll() is not None: raise ValueError('worker exited before identity handshake; lane remains held')
            time.sleep(.05)
        for line in notes: print(line)
        label='resume '+session[:8] if session else 'new session'
        print(f'dispatched {out.name} on lane {lane} [{self.role(lane)}] (danger-full-access, {model}, {label}) pid {current["bash_pid"]}')
        return 0

    def worker(self, run, bash_pid):
        run=Path(run); job=rt.read_json(run/'job.json'); lane=job['lane']
        with self.mutex():
            current=rt.read_json(self.file(lane,'state.json'))
            if current['run_id']!=job['run_id'] or current['phase']!='reserved':
                raise ValueError('reservation identity differs')
            job.update(phase='running',worker=rt.identity(),bash_pid=int(bash_pid),
                       containment=contain_worker(run))
            rt.atomic_text(self.file(lane,'lock'),str(bash_pid)+'\n')
            rt.atomic_json(self.file(lane,'state.json'),job); rt.atomic_json(run/'job.json',job)
            self.registry()
        report=run/'report.md'; log=Path(job['log'])
        common=['-m',job['model'],'-c','model_reasoning_effort='+job['effort'],
                '-c','model_auto_compact_token_limit=80000','-c','sandbox_mode="danger-full-access"','--skip-git-repo-check']
        if job['nosubagents']: common+=['--disable','multi_agent']
        common+=['--json','-o',str(report)]
        args=['exec','resume',job['session'],*common,'-'] if job['session'] else ['exec',*common,'-s','danger-full-access','-C','.','-']
        # A script avoids Windows CRT/MSYS disagreement about embedded quotes
        # in TOML values, while preserving every original Codex argument.
        rt.atomic_text(run/'command.sh', '#!/usr/bin/env bash\nexec '+shlex.join(['codex', *args])+'\n')
        code=127
        try:
            with (run/'prompt.txt').open('rb') as inp, log.open('wb') as output:
                child=subprocess.Popen([bash(),str(run/'command.sh')],cwd=self.root,
                                       stdin=inp,stdout=output,stderr=subprocess.STDOUT,**launch_options())
                job['child']=rt.identity(child.pid)
                with self.mutex():
                    rt.atomic_json(run/'job.json',job); rt.atomic_json(self.file(lane,'state.json'),job)
                code=child.wait()
        except OSError as exc:
            with log.open('ab') as output: output.write(('DISPATCH_ERROR '+str(exc)+'\n').encode())
        shutil.copyfile(log,run/'run.log')
        session=job['session']
        if not session:
            with log.open(encoding='utf-8',errors='replace') as stream:
                for line in stream:
                    try: event=json.loads(line)
                    except ValueError: continue
                    candidate=next((event[k] for k in ('thread_id','session_id','conversation_id') if k in event),None) if isinstance(event,dict) else None
                    if isinstance(candidate,str) and re.fullmatch(r'[0-9a-fA-F-]{36}',candidate): session=candidate; break
        success=code==0 and report.is_file() and report.stat().st_size>0
        receipt=dict(schema='dispatch-terminal/v1',run_id=job['run_id'],lane=lane,out=job['out'],
                     state='SUCCESS' if success else 'FAILED',exit_code=code,
                     report_sha256=sha(report) if report.exists() else None,log_sha256=sha(run/'run.log'),
                     prompt_sha256=job['prompt_sha256'],worker=job['worker'],child=job['child'],
                     sources=job['sources'],completed=dt.datetime.now().isoformat())
        with self.mutex():
            if rt.read_json(self.file(lane,'state.json'))['run_id']!=job['run_id']:
                raise ValueError('lane identity changed during execution')
            if success:
                published=Path(job['out']+'.publishing-'+job['run_id'])
                shutil.copyfile(report,published)
                rt.replace_file(published,job['out'])
            rt.atomic_json(run/'receipt.json',receipt)
            rt.atomic_json(job['out']+'.receipt.json',receipt)
            if session:
                rt.atomic_text(self.file(lane,'id'),session+'\n')
                if not job['session']: rt.atomic_text(self.file(lane,'opened'),stamp()+'\n')
            rt.atomic_text(self.file(lane,'n'),str(job['count']+int(success))+'\n')
            job['phase']='terminal'
            rt.atomic_json(run/'job.json',job); rt.atomic_json(self.file(lane,'state.json'),job)
            self.file(lane,'lock').unlink(missing_ok=True)
            metadata = overflow_metadata(job)
            if metadata and not metadata['keep_session']:
                self.clear_session(lane)
            self.registry()
        return 0 if success else 1

    def wait(self, out_arg, lane=''):
        if lane: lane = canonical_lane(lane, notice=True)
        out=Path(out_arg); out=out if out.is_absolute() else self.root/out
        pointer=Path(str(out)+'.dispatch.json')
        expected=None
        while True:
            try:
                if pointer.exists():
                    point=rt.read_json(pointer)
                    if expected is None: expected=point
                    if point!=expected: break
                    run=Path(point['run']); job=rt.read_json(run/'job.json')
                    if job['run_id']!=point['run_id'] or job['out']!=str(out.resolve()): break
                    if job['lane'] != point['lane']: break
                    if lane and canonical_lane(point['lane']) != lane:
                        metadata = overflow_metadata(job)
                        if not metadata or canonical_lane(metadata['requested_lane']) != lane: break
                    if job.get('release'): break
                    receipt_file=run/'receipt.json'
                    if receipt_file.exists() and job['phase']=='terminal':
                        receipt=rt.read_json(receipt_file)
                        valid=(receipt.get('schema')=='dispatch-terminal/v1' and receipt['run_id']==point['run_id'] and
                               receipt['lane']==point['lane'] and receipt['out']==str(out.resolve()) and
                               receipt['state']=='SUCCESS' and type(receipt['exit_code']) is int and receipt['exit_code']==0 and
                               receipt['worker']==job['worker'] and receipt['child']==job['child'] and
                               receipt['prompt_sha256']==sha(run/'prompt.txt') and receipt['sources']==job['sources'] and
                               receipt['log_sha256']==sha(run/'run.log') and
                               receipt['report_sha256']==sha(out)==sha(run/'report.md') and
                               rt.read_json(str(out)+'.receipt.json')==receipt and rt.liveness(job['child'])=='dead')
                        if not valid: break
                        print(f'DONE {dt.datetime.now():%H:%M} {out_arg}',flush=True)
                        with out.open('rb') as stream: print(stream.read(1500).decode('utf-8',errors='replace'),end='')
                        return 0
                    if job['phase']=='running' and rt.liveness(job['worker'])=='dead':
                        if rt.read_json(run/'job.json')['phase']=='terminal': continue
                        break
                    if job['phase']=='reserved' and rt.liveness(job['owner'])=='dead':
                        if rt.read_json(run/'job.json')['phase']!='reserved': continue
                        break
                else:
                    # Existing legacy jobs have no terminal provenance. Wait for
                    # an active lane, then report failure; report presence alone
                    # can never establish successful completion.
                    with self.mutex():
                        if not lane or not self.busy(lane): break
            except (OSError,ValueError,KeyError,TypeError):
                break
            time.sleep(.25)
        print(f'LANE_ENDED_WITHOUT_REPORT {dt.datetime.now():%H:%M} {lane} {out_arg}')
        return 1


def main(args=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    args=sys.argv[1:] if args is None else args
    try:
        if args and args[0]=='_guardian': return tree_guardian(args[1])
        if args and args[0]=='_worker':
            return Lanes(args[1]).worker(args[2],args[3])
        runtime=Lanes(os.environ.get('KNEE_WORKSPACE_ROOT',rt.CODE_ROOT))
        if args and args[0]=='_wait': return runtime.wait(args[1],args[2] if len(args)>2 else '')
        if not args: raise ValueError('lane must be one of: '+' '.join(LANES))
        return runtime.dispatch(args)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        print(str(exc)); return 2


if __name__=='__main__':
    raise SystemExit(main())
