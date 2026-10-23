import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("input_gate",Path(__file__).resolve().parents[1]/"tools/d01_input_gate.py")
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def test_reference_constant_and_known_quarter_coordinate_impulse():
    volume=np.zeros((3,336,336),np.uint8)
    volume[:,0,0]=255
    windows,_=gate.reference(volume,[1])
    assert windows.shape==(1,3,224,224)
    np.testing.assert_array_equal(windows[0,:,0,0],np.full(3,.75*.75))
    assert np.count_nonzero(windows)==3
    windows,_=gate.reference(np.full_like(volume,255),[1])
    np.testing.assert_array_equal(windows,np.ones_like(windows))


@pytest.mark.parametrize("bad",[np.zeros((3,336,336),np.float32),np.zeros((3,224,224),np.uint8)])
def test_reference_rejects_contract_drift(bad):
    with pytest.raises(ValueError):gate.reference(bad,[1])


def test_elementwise_absolute_only_gate_rejects_nan_shape_dtype_and_one_outlier():
    expected=np.zeros(100,np.float64);actual=np.zeros(100,np.float32)
    actual[0]=2**-10
    assert gate.compare(actual,expected,2**-10)["pass"]
    assert not gate.compare(actual,expected,2**-11)["pass"]
    actual[0]=np.nan
    assert not gate.compare(actual,expected,1)["pass"]
    assert not gate.compare(np.zeros(99,np.float32),expected,1)["pass"]
    assert not gate.compare(np.zeros(100,np.float64),expected,1)["pass"]


def test_integer_input_change_fails_before_float_comparison(monkeypatch):
    volume=np.zeros((3,336,336),np.uint8);mask=np.ones(3,np.uint8)
    expected=gate.expected_row("case",volume,mask,[1]);mean,std=gate.constants()
    protocol={"mean_sha256":gate.array_sha(mean),"std_sha256":gate.array_sha(std)}
    def forbidden(*args):raise AssertionError("float path must not execute")
    monkeypatch.setitem(sys.modules,"raptor_common",SimpleNamespace(windows_from=forbidden))
    monkeypatch.setitem(sys.modules,"raptor_runtime",SimpleNamespace(normalize_input=forbidden,IMAGENET_MEAN=mean,IMAGENET_STD=std))
    volume[1,0,0]=1
    result=gate.check_case(volume,mask,[1],expected,protocol)
    assert result["pass"] is False and result["exact_input"] is False


def test_seven_synthetic_inputs_reproduce_without_metric_or_labels():
    a=list(gate.synthetic_cases());b=list(gate.synthetic_cases())
    assert len(a)==7 and len({x[0] for x in a})==7
    for x,y in zip(a,b):
        np.testing.assert_array_equal(x[1],y[1])
