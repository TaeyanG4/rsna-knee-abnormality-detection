"""No-network regression tests for download secrecy and immutable destinations."""
import builtins
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import traceback

import pytest

spec = importlib.util.spec_from_file_location(
    "kaggle_output_file_under_test", Path(__file__).resolve().parents[1] / "tools/kaggle_output_file.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)
SECRET = "https://private.example/output?token=TOKEN_SENTINEL_MUST_NEVER_LEAK"


class FakeHttpError(RuntimeError):
    def __init__(self):
        super().__init__(SECRET)
        self.response = SimpleNamespace(status_code=403)


def install_clients(monkeypatch, *, fail=None, headers=None, chunks=None, status=200, on_stream=None):
    calls = []
    def fail_at(stage):
        calls.append(stage)
        if fail == stage:
            print(SECRET)
            print(SECRET, file=sys.stderr)
            try:
                raise ValueError(SECRET)
            except ValueError as cause:
                raise FakeHttpError() from cause
        if fail == "auth_exit" and stage == "auth":
            print(SECRET, file=sys.stderr)
            raise SystemExit(SECRET)

    class Response:
        status_code = status
        def __init__(self):
            self.headers = {} if headers is None else headers
        def __enter__(self):
            return self
        def __exit__(self, *args):
            fail_at("http_exit")
        def iter_content(self, size):
            for chunk in chunks or [b"first", b"second"]:
                yield chunk
                if on_stream:
                    on_stream()
                fail_at("stream")

    def redirect(request):
        fail_at("redirect")
        assert (request.owner_slug, request.kernel_slug, request.version_number, request.file_path) == ("owner", "kernel", 2, "artifact.json")
        return SimpleNamespace(url=SECRET)
    class Api:
        def __init__(self):
            fail_at("constructor")
        def authenticate(self):
            fail_at("auth")
        def build_kaggle_client(self):
            fail_at("client")
            return nullcontext(SimpleNamespace(kernels=SimpleNamespace(
                kernels_api_client=SimpleNamespace(download_kernel_output=redirect))))
    def get(url, **kwargs):
        fail_at("http")
        assert url == SECRET and kwargs == {"stream": True, "timeout": (15, 60)}
        return Response()

    for name in ["kaggle", "kaggle.api", "kaggle.api.kaggle_api_extended", "kagglesdk",
                 "kagglesdk.kernels", "kagglesdk.kernels.types",
                 "kagglesdk.kernels.types.kernels_api_service", "requests"]:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["kaggle.api.kaggle_api_extended"].KaggleApi = Api
    sys.modules["kagglesdk.kernels.types.kernels_api_service"].ApiDownloadKernelOutputRequest = SimpleNamespace
    sys.modules["requests"].get = get
    return calls


def invoke(out, **kwargs):
    return downloader.download("owner/kernel", 2, "artifact.json", out, **kwargs)


def assert_sanitized(error, capsys, receipt):
    captured = capsys.readouterr()
    text = "".join(traceback.format_exception(error)) + captured.out + captured.err + receipt.read_text()
    assert SECRET not in text and "TOKEN_SENTINEL" not in text
    assert error.__context__ is None and error.__cause__ is None
    record = json.loads(receipt.read_text())
    assert record["status"] == "FAILED"
    assert set(record) == {"observed_at", "kernel", "version", "file", "local_name", "max_bytes",
                           "status", "error_type", "http_status", "stage"}
    return record


@pytest.mark.parametrize("failure,stage", [
    ("constructor", "authentication"), ("auth", "authentication"), ("auth_exit", "authentication"),
    ("client", "sdk_redirect"), ("redirect", "sdk_redirect"), ("http", "http_request"),
    ("stream", "download_stream"), ("http_exit", "download_stream")])
def test_external_error_and_direct_print_never_leak(tmp_path, monkeypatch, capsys, failure, stage):
    install_clients(monkeypatch, fail=failure)
    out = tmp_path / "out.json"
    with pytest.raises(RuntimeError) as caught:
        invoke(out)
    record = assert_sanitized(caught.value, capsys, tmp_path / "out.json.download.json")
    assert record["stage"] == stage
    assert record["http_status"] == (None if failure == "auth_exit" else 403)
    assert not out.exists()
    if failure == "stream":
        assert (tmp_path / "out.json.part").read_bytes() == b"first"


def test_import_error_is_sanitized(tmp_path, monkeypatch, capsys):
    install_clients(monkeypatch)
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name == "kaggle.api.kaggle_api_extended":
            print(SECRET)
            raise ImportError(SECRET)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(RuntimeError) as caught:
        invoke(tmp_path / "out")
    record = assert_sanitized(caught.value, capsys, tmp_path / "out.download.json")
    assert record["stage"] == "imports" and record["error_type"] == "ImportError"


def test_invalid_header_external_value_is_not_echoed(tmp_path, monkeypatch, capsys):
    install_clients(monkeypatch, headers={"Content-Length": SECRET})
    with pytest.raises(RuntimeError) as caught:
        invoke(tmp_path / "out")
    record = assert_sanitized(caught.value, capsys, tmp_path / "out.download.json")
    assert record["stage"] == "size_header"


def test_http_status_is_a_safe_integer(tmp_path, monkeypatch, capsys):
    install_clients(monkeypatch, status=404)
    with pytest.raises(RuntimeError) as caught:
        invoke(tmp_path / "out")
    record = assert_sanitized(caught.value, capsys, tmp_path / "out.download.json")
    assert record["http_status"] == 404


def test_success_sha_and_size(tmp_path, monkeypatch, capsys):
    install_clients(monkeypatch, headers={"Content-Length": "11"})
    out = tmp_path / "out"
    digest = hashlib.sha256(b"firstsecond").hexdigest()
    result = invoke(out, expected_sha256=digest, max_bytes=11)
    assert result["status"] == "DOWNLOADED" and result["bytes"] == 11 and result["sha256"] == digest
    assert out.read_bytes() == b"firstsecond" and not (tmp_path / "out.part").exists()
    assert json.loads((tmp_path / "out.download.json").read_text()) == result
    captured = capsys.readouterr()
    assert not captured.out and not captured.err


@pytest.mark.parametrize("mode,stage,partial", [
    ("declared", "size_header", None), ("actual", "download_stream", b"first"),
    ("hash", "sha256_validation", b"firstsecond")])
def test_size_and_hash_fail_closed(tmp_path, monkeypatch, capsys, mode, stage, partial):
    install_clients(monkeypatch, headers={"Content-Length": "100"} if mode == "declared" else {})
    with pytest.raises(RuntimeError) as caught:
        invoke(tmp_path / "out", max_bytes=5 if mode != "hash" else 100,
               expected_sha256="0" * 64 if mode == "hash" else None)
    record = assert_sanitized(caught.value, capsys, tmp_path / "out.download.json")
    assert record["stage"] == stage and not (tmp_path / "out").exists()
    if partial is None:
        assert not (tmp_path / "out.part").exists()
    else:
        assert (tmp_path / "out.part").read_bytes() == partial


@pytest.mark.parametrize("name", ["out", "out.part", "out.download.json"])
def test_existing_destination_blocks_before_auth(tmp_path, monkeypatch, name):
    calls = install_clients(monkeypatch)
    existing = tmp_path / name
    existing.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        invoke(tmp_path / "out")
    assert existing.read_bytes() == b"preserve" and not calls


def test_concurrent_receipt_is_preserved_without_original_exception_context(tmp_path, monkeypatch, capsys):
    receipt = tmp_path / "out.download.json"
    install_clients(monkeypatch, fail="stream", on_stream=lambda: receipt.write_bytes(b"other run"))
    with pytest.raises(FileExistsError) as caught:
        invoke(tmp_path / "out")
    captured = capsys.readouterr()
    text = "".join(traceback.format_exception(caught.value)) + captured.out + captured.err
    assert SECRET not in text and "TOKEN_SENTINEL" not in text
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert receipt.read_bytes() == b"other run" and (tmp_path / "out.part").read_bytes() == b"first"


def test_destination_created_during_stream_is_preserved(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    install_clients(monkeypatch, on_stream=lambda: out.write_bytes(b"other run"))
    with pytest.raises(RuntimeError) as caught:
        invoke(out)
    record = assert_sanitized(caught.value, capsys, tmp_path / "out.download.json")
    assert record["stage"] == "publish_output"
    assert out.read_bytes() == b"other run"
    assert (tmp_path / "out.part").read_bytes() == b"firstsecond"


def test_receipt_write_failure_does_not_resurrect_external_exception(tmp_path, monkeypatch, capsys):
    install_clients(monkeypatch, fail="redirect")
    def cannot_write(*args):
        raise OSError("local receipt unavailable")
    monkeypatch.setattr(downloader, "_write_receipt", cannot_write)
    with pytest.raises(OSError) as caught:
        invoke(tmp_path / "out")
    captured = capsys.readouterr()
    assert SECRET not in "".join(traceback.format_exception(caught.value)) + captured.out + captured.err
    assert caught.value.__context__ is None and caught.value.__cause__ is None
