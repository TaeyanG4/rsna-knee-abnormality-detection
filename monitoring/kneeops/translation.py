"""Reviewed-in-workspace translations pinned to exact public source text.

No runtime translation service, API key or outbound content upload is involved.
An updated source must never silently inherit the previous version's translation.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parent / 'translations'


@lru_cache(maxsize=1)
def catalog():
    items = json.loads((ROOT / 'catalog.json').read_text(encoding='utf-8'))
    result = []
    for item in items:
        slug = item['slug']
        if not slug or any(c not in 'abcdefghijklmnopqrstuvwxyz-' for c in slug):
            raise ValueError('Invalid translation catalog slug')
        original = (ROOT / 'en' / f'{slug}.md').read_text(encoding='utf-8')
        if hashlib.sha256(original.encode('utf-8')).hexdigest() != item['source_sha']:
            raise ValueError('Translation source checksum mismatch: ' + slug)
        result.append({**item, 'content': (ROOT / 'ko' / f'{slug}.md').read_text(encoding='utf-8')})
    return result


def translation_for(source):
    if source['kind'] != 'official':
        return None
    page = source['title'].split(' · ')[-1].casefold()
    candidates = [item for item in catalog() if item['competition'] == source['competition'] and item['page'].casefold() == page]
    for item in candidates:
        if item['source_sha'] == source['revision']['sha']:
            return {**item, 'status': 'available', 'language': 'ko', 'disclaimer': '이 한국어판은 AI 참고 번역입니다. 규칙 해석의 기준은 영어 원문입니다.'}
    return {'status': 'outdated' if candidates else 'unavailable', 'language': 'ko',
            'message': '원문이 변경되어 이 버전의 번역을 다시 확인해야 합니다. 이전 저장본을 선택하면 해당 번역을 볼 수 있습니다.' if candidates else '이 문서의 한국어 번역은 아직 준비되지 않았습니다.',
            'source_sha': source['revision']['sha']}
