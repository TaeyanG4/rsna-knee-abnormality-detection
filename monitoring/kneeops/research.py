"""Literal strategy excerpts, not model-generated or verified performance claims."""
import re

ASPECTS = {
    '검증 설계': r'cross.?val|\bfolds?\b|\boof\b|patient.?level|groupkfold',
    '입력·전처리': r'dicom|resampl|normaliz|localiz|crop|slice select|2\.5d',
    '모델 구조': r'backbone|convnext|efficientnet|transformer|attention|\b3d\b|\b2d\b|\bvit\b',
    '라벨·학습': r'pseudo.?label|distill|mixup|loss|augment|class weight',
    '앙상블': r'ensemble|blend|model averaging|\btta\b',
    '추론 효율': r'inference|runtime|quantiz|onnx|tensorrt',
}


def strategy_excerpts(content):
    excerpts=[]
    lines=content.splitlines()
    for aspect,pattern in ASPECTS.items():
        hits=0
        for index,line in enumerate(lines):
            if len(line.strip())>=16 and re.search(pattern,line,re.I):
                excerpts.append({'aspect':aspect,'line':index+1,'quote':line.strip()[:600]})
                hits+=1
                if hits==2:break
    return excerpts


DEFAULT_COMPETITIONS = [
    {'slug':'rsna-intracranial-aneurysm-detection','title':'RSNA 2025 · Intracranial Aneurysm','relevance':'가장 최근 RSNA 대회. CTA·MRA·MRI가 섞인 다중 모달리티 시리즈, 동맥류 존재(가중치 13)와 13개 위치의 AUC를 반씩 반영하는 14-label 가중 AUC, evaluation API 노트북 제출을 참고. 혈관 검출 과제이므로 무릎 소견과 해부학이 다릅니다.'},
    {'slug':'rsna-2024-lumbar-spine-degenerative-classification','title':'RSNA 2024 · Lumbar Spine MRI','relevance':'다중 MRI 시리즈, 위치 검출과 단계별 분류를 참고. 척추 퇴행성 분류와 무릎 12-target AUC는 평가 방식이 다릅니다.'},
    {'slug':'rsna-2023-abdominal-trauma-detection','title':'RSNA 2023 · Abdominal Trauma','relevance':'3D 의료 영상의 슬라이스 선택·앙상블·실행 시간 전략을 참고. CT 대회이므로 MRI 전처리로 그대로 옮기지 않습니다.'},
    {'slug':'rsna-2022-cervical-spine-fracture-detection','title':'RSNA 2022 · Cervical Spine Fracture','relevance':'위치 검출과 분류를 분리하는 접근을 참고. CT·척추 골절 과제와 현재 대회의 차이를 확인하세요.'},
    {'slug':'rsna-miccai-brain-tumor-radiogenomic-classification','title':'RSNA-MICCAI 2021 · Brain Tumor MRI','relevance':'FLAIR·T1w·T1wCE·T2w 다중 시퀀스 MRI의 study 단위 AUC 분류. 약한 신호와 큰 Public/Private 순위 변동 사례로 검증 설계를 참고. 뇌종양 유전형 과제라 무릎 소견과 직접 비교하지 않습니다.'},
]
