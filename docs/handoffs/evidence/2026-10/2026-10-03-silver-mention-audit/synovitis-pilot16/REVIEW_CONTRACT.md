# Synovitis 가림16 검토 계약
- blinded-packet.json만 읽고 각 case_id에 full Report 근거를 기록한다. sealed는 판정 저장·SHA256 동결 전 접근하지 않는다.
- 상태: explicit positive / explicit negative / uncertain / proxy-only / unmentioned / conflicting.
- explicit positive에는 명시적 synovitis 진단 또는 직접적인 synovial thickening/hypertrophy/inflammation 문구가 필요하다.
- 단순 effusion, plica, Hoffa 표현만으로 자동양성 판정하지 않는다. 직접근거 없이 이런 표현만 있으면 proxy-only 등 불확실성을 보존한다.
- 미언급은 영상음성이 아니다. 인용구와 판단에 필요한 정보의 부재를 보존한다.
- 이 표적16건은 소스감사이며 임상정답 생성·오류율 추정·성능평가가 아니다. 판정 후에도 자동 라벨교체·마스킹·학습·확대를 하지 않는다.
- 이 파일은 향후 검토 계약이며 builder가 검토를 수행했다는 뜻이 아니다.
