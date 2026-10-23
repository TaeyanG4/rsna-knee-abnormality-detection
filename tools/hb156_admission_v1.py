"""HB151 seed44 admission v1.0.0: refuse busy GPU immediately; never wait.

Changelog: process-table fallback for container nvidia-smi PID omissions
and fresh memory admission.
"""
import os
import psutil


def hb151_processes(allowed=()):
    blocked = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.pid in allowed:
            continue
        try:
            info = process.info
            if 'python' not in (info.get('name') or '').lower():
                continue
            args = info.get('cmdline')
            if args is None:
                raise ValueError('cannot inspect Python process for HB151 ownership')
            modules = [arg for arg in args if any(tag in arg.lower() for tag in ('hb151','hb156')) and
                       (arg.startswith('tools.') or arg.endswith('.py'))]
            if modules and any(arg in ('fit', '_fit', 'smoke', '_smoke', 'train') for arg in args):
                blocked.append(process.pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as exc:
            raise ValueError('cannot inspect process ownership; refusing GPU') from exc
    return blocked


def require_idle(memory_snapshot):
    allowed = [os.getpid()] + [p.pid for p in psutil.Process().parents()]
    blocked = hb151_processes(allowed)
    if blocked:
        raise ValueError('another HB151/HB156 fit/smoke holds GPU; refusing start: ' + str(blocked))
    memory = memory_snapshot()
    # The inherited smoke ceiling is 12 GiB; retain 2 GiB free headroom.
    if memory['free_mib'] < 14 * 1024:
        raise ValueError('GPU busy: less than 14 GiB free; refusing start')
    return {'hb151_blockers': [], 'free_mib': memory['free_mib']}
