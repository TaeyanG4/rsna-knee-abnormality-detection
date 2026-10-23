import random
import numpy as np
import pytest
import torch
from tools import pl_production_loader_v1 as l


def test_worker_seed_covers_python_numpy_torch(monkeypatch):
    monkeypatch.setattr(torch, 'initial_seed', lambda: 123)
    l.seed_worker(0)
    a=(random.random(),np.random.rand(),torch.rand(1).item())
    l.seed_worker(5)
    assert a==(random.random(),np.random.rand(),torch.rand(1).item())


def test_defaults_and_invalid(monkeypatch):
    monkeypatch.setattr(l.os,'cpu_count',lambda:16)
    if hasattr(l.os,'sched_getaffinity'):
        monkeypatch.setattr(l.os,'sched_getaffinity',lambda _:set(range(16)))
    assert l.resolve_workers(None)==l.resolve_workers(None,'vast')==6
    for n in (-1,16,1.5):
        with pytest.raises(ValueError):l.resolve_workers(n)
    for args in ((6,5,40),(6,2,41),(6,2,1)):
        with pytest.raises(ValueError):l.memory_plan(*args)


def test_generators_are_separate_and_cleanup_on_error():
    ds=torch.utils.data.TensorDataset(torch.arange(16))
    loader,gen,cfg=l.make_loader(ds,None,42,workers=0)
    assert gen is not loader.generator
    assert not cfg['persistent_workers'] and not cfg['pin_memory']
    assert cfg['estimated_bytes']<40e9
    gen.manual_seed(43);a=list(loader)
    gen.manual_seed(43);b=list(loader)
    assert all(torch.equal(x[0],y[0]) for x,y in zip(a,b))


def test_managed_loader_shuts_workers_down_on_exception():
    from types import SimpleNamespace
    seen=[]
    loader=SimpleNamespace(_iterator=SimpleNamespace(_shutdown_workers=lambda:seen.append(1)))
    with pytest.raises(RuntimeError),l.managed_loader(loader):
        raise RuntimeError('abort')
    assert seen==[1]


def test_cuda_configuration_without_initializing_cuda():
    ds=torch.utils.data.TensorDataset(torch.arange(16))
    loader,_,cfg=l.make_loader(ds,None,42,workers=2,device='cuda')
    assert loader.pin_memory and loader.persistent_workers
    assert loader.prefetch_factor==2 and loader.worker_init_fn is l.seed_worker
    assert loader.multiprocessing_context.get_start_method()=='spawn'
    assert cfg['estimated_bytes']<=cfg['ram_budget_bytes']==40_000_000_000
    assert not torch.cuda.is_initialized()
