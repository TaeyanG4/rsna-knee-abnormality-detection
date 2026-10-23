"""Download one explicit Kaggle notebook output at an exact version.

No output-directory enumeration, account mutation, credential or signed-URL log.
Run with the app .venv. Existing destinations are never overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def _http_status(error):
    """Read only a bounded integer; never stringify an external exception."""
    try:
        value = getattr(getattr(error, 'response', None), 'status_code', None)
        return value if type(value) is int and 100 <= value <= 599 else None
    except Exception:
        return None


def _write_receipt(path, record):
    # A destination created by another process must also remain untouched.
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(record, indent=2)+'\n')


def _quiet_dependencies(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # Authentication/import failures may print before raising. CLI JSON is
        # printed only after download returns, outside this dependency boundary.
        with open(os.devnull, 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
            return function(*args, **kwargs)
    return wrapped


@_quiet_dependencies
def download(kernel, version, file, out, max_bytes=16 << 20, expected_sha256=None):
    owner, slug = kernel.split('/')
    name = PurePosixPath(file)
    if not owner or not slug or version < 1 or name.is_absolute() or '..' in name.parts or '\\' in file:
        raise ValueError('Explicit owner/slug, positive version and relative output file required')
    if max_bytes < 1:
        raise ValueError('Positive download size bound required')
    out = Path(out)
    partial = out.with_name(out.name+'.part')
    receipt = out.with_name(out.name+'.download.json')
    if any(p.exists() for p in (out, partial, receipt)):
        raise FileExistsError('Choose a new destination; prior outputs/partials are preserved')
    record = dict(observed_at=datetime.now(timezone.utc).isoformat(), kernel=kernel,
                  version=version, file=file, local_name=out.name, max_bytes=max_bytes)
    out.parent.mkdir(parents=True, exist_ok=True)
    stage, http_status, failed = 'imports', None, False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
        import requests
        stage = 'authentication'
        api = KaggleApi(); api.authenticate()
        stage = 'sdk_redirect'
        request = ApiDownloadKernelOutputRequest()
        request.owner_slug = owner; request.kernel_slug = slug
        request.version_number = version; request.file_path = file
        with api.build_kaggle_client() as client:
            redirect = client.kernels.kernels_api_client.download_kernel_output(request)
        digest = hashlib.sha256(); size = 0
        stage = 'http_request'
        with requests.get(redirect.url, stream=True, timeout=(15, 60)) as response:
            value = response.status_code
            http_status = value if type(value) is int and 100 <= value <= 599 else None
            if http_status != 200:
                raise RuntimeError('Unexpected download HTTP status')
            stage = 'size_header'
            declared = response.headers.get('Content-Length')
            if declared and int(declared) > max_bytes:
                raise ValueError('Output exceeds declared download size bound')
            stage = 'download_stream'
            with partial.open('xb') as target:
                for chunk in response.iter_content(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError('Output exceeds actual download size bound')
                    digest.update(chunk); target.write(chunk)
        stage = 'sha256_validation'
        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            raise ValueError('Downloaded file SHA256 mismatch; partial preserved')
        # On Windows rename refuses an existing destination. Check also for POSIX.
        stage = 'publish_output'
        if out.exists():
            raise FileExistsError('Destination appeared while downloading')
        partial.rename(out)
        record.update(status='DOWNLOADED', bytes=size, sha256=actual)
    except (Exception, SystemExit) as exc:
        error_type = type(exc).__name__
        if not error_type.isascii() or not error_type.isidentifier() or len(error_type) > 80:
            error_type = 'ExternalError'
        record.update(status='FAILED', error_type=error_type,
                      http_status=_http_status(exc) or http_status, stage=stage)
        failed = True
    _write_receipt(receipt, record)
    if failed:
        # Raise outside except: even __context__ must not retain a signed URL.
        raise RuntimeError(f"Output download failed at {record['stage']} ({record['error_type']}); see {receipt.name}") from None
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernel', required=True); p.add_argument('--version', type=int, required=True)
    p.add_argument('--file', required=True); p.add_argument('--out', required=True)
    p.add_argument('--max-bytes', type=int, default=16 << 20)
    p.add_argument('--expected-sha256')
    print(json.dumps(download(**vars(p.parse_args()))))
