"""Local viewer routes share the app's origin and CSRF protection."""
from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .store import TARGETS
from .viewer import Viewer


class OpenImage(BaseModel):
    source: str = Field(default='', max_length=100)
    uid: str = Field(default='', max_length=100)
    path: str = Field(default='', max_length=1000)
    raw: str = Field(default='', max_length=24)


class Frame(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    session: str = Field(pattern=r'^[a-f0-9]{32}$')
    slot: int = Field(default=0, ge=0, le=1000)
    slice: int = Field(default=0, ge=0, le=4095)
    low: float | None = None
    high: float | None = None
    invert: bool = False
    roi: list[float] | None = Field(default=None, min_length=4, max_length=4)


class Review(Frame):
    note: str = Field(default='', max_length=4000)
    flags: list[Literal['checked', 'motion', 'crop', 'missing', 'label_check']] = Field(default_factory=list, max_length=5)


class Audit(BaseModel):
    session: str = Field(pattern=r'^[a-f0-9]{32}$')
    slot: int = Field(default=0, ge=0, le=1000)
    samples: int = Field(default=16, ge=3, le=64)


class Preview(Frame):
    resolution: Literal[128, 224, 256, 288, 336, 384, 512] = 288
    crop_percent: int = Field(default=100, ge=25, le=100)
    normalization: Literal['minmax', 'p1', 'p2'] = 'p1'


def viewer_router(store):
    viewer = Viewer(store)
    router = APIRouter(prefix='/api/viewer')

    @router.get('/catalog')
    def catalog():
        return viewer.catalog()

    @router.get('/browse')
    def browse(root: str, relative: str = Query('', max_length=1000)):
        return viewer.browse(root, relative)

    @router.get('/studies')
    def studies(source: str, q: str = Query('', max_length=100),
                mode: Literal['all', 'missing', 'labeled'] = 'all', offset: int = Query(0, ge=0), target: str = '',
                split: Literal['', 'train', 'test'] = '', root: str = Query('', max_length=1000)):
        if target and target not in TARGETS:
            raise ValueError('알 수 없는 소견입니다.')
        return viewer.studies(source, q, mode, offset, target, split, root)

    @router.get('/tree/series')
    def tree_series(source: str, uid: str = Query(max_length=100)):
        index = viewer.cache_index(viewer.cache_path(source))
        if uid not in index['uids']:
            raise ValueError('캐시에 해당 study가 없습니다.')
        i = index['uids'].index(uid)
        m = index['manifest']; names = m.get('slots', [])
        return {'series': [{'slot': s, 'name': (names[s].get('name', str(s)) if isinstance(names[s], dict) else names[s]) if s < len(names) else f'Slot {s+1}',
                            'valid': index['mask'][i, s].astype(bool).tolist(), 'slices': m['shape_total'][2]}
                           for s in range(m['shape_total'][1])]}

    @router.get('/eda/dataset')
    def eda_dataset():
        from .viewer_eda import dataset_summary
        return dataset_summary(viewer)

    @router.post('/eda/series')
    def eda_series(body: Audit):
        from .viewer_eda import series_audit
        return series_audit(viewer, body.session, body.slot, body.samples)

    @router.post('/eda/preview')
    def eda_preview(body: Preview):
        from .viewer_eda import preprocessing_preview
        return preprocessing_preview(viewer, body.session, body.slot, body.slice, body.resolution,
                                     body.crop_percent, body.normalization)

    @router.post('/open')
    def open_image(body: OpenImage):
        if sum(map(bool, (body.source, body.path, body.raw))) != 1:
            raise ValueError('캐시, 로컬 원본 또는 경로 하나를 선택하세요.')
        if body.raw:
            from .data_library import study_path
            folder, item = study_path(store, body.raw)
            opened = viewer.open(path=str(folder))
            opened['raw_id'] = body.raw
            opened['local_inventory'] = item
            if item['status'] != 'complete':
                opened['warnings'].insert(0, '부분 다운로드 또는 파일 완전성 미확인 study입니다. 학습 입력으로 사용하지 마세요.')
            return opened
        return viewer.open(body.source, body.uid, body.path)

    @router.post('/frame')
    def frame(body: Frame):
        return viewer.frame(body.session, body.slot, body.slice, body.low, body.high, body.invert, body.roi)

    @router.get('/contact')
    def contact(session: str, slot: int = Query(0, ge=0)):
        return viewer.contact(session, slot)

    @router.post('/reviews')
    def review(body: Review):
        return viewer.review(body.session, body.slot, body.slice, body.note, body.flags, body.roi,
                             body.low, body.high, body.invert)

    @router.get('/export')
    def export(session: str):
        item = viewer.session(session)
        return JSONResponse({'schema': 1, 'source': viewer.uri(item['path']), 'uid': item['uid'],
                             'reviews': viewer.reviews(item)},
                            headers={'Content-Disposition': 'attachment; filename="kneeops-image-review.json"'})

    return router
