"""Isolated, read-only SDK call. CLI pull omits the server version metadata."""
import json
import re
import sys


def main():
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    ref=sys.argv[1]
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[1-9][0-9]*)?',ref):raise ValueError('Invalid notebook ref')
    parts=ref.split('/')
    request=ApiGetKernelRequest()
    request.user_name=parts[0];request.kernel_slug='/'.join(parts[1:])
    with api.build_kaggle_client() as client:
        response=client.kernels.kernels_api_client.get_kernel(request)
    meta=response.metadata;blob=response.blob
    if len(blob.source.encode('utf-8'))>10_000_000:raise ValueError('Source exceeds 10 MB')
    # Preserve response metadata separately from the requested version.
    print(json.dumps({'ref':'/'.join(parts[:2]),'title':meta.title,'source':blob.source,'kernel_type':blob.kernel_type,
        'requested_version':parts[2] if len(parts)==3 else None,'server_current_version':meta.current_version_number,
        'version':parts[2] if len(parts)==3 else str(meta.current_version_number or 'unknown'),
        'inputs':meta.dataset_data_sources,'kernel_inputs':meta.kernel_data_sources,'competition_inputs':meta.competition_data_sources,
        'model_inputs':meta.model_data_sources},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        message=str(exc)
        if '403' in message:message='Kaggle에서 이 버전의 접근을 허용하지 않았습니다 (403). 공개된 버전만 수집할 수 있습니다.'
        elif '404' in message:message='Kaggle에서 이 노트북 버전을 찾을 수 없습니다 (404). 버전 번호와 공개 여부를 확인하세요.'
        print(message[:700],file=sys.stderr)
        sys.exit(1)
