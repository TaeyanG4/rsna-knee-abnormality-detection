# Synovitis16 가림 대조 결과

2026-10-03 23:57 KST. 두 LLM 가림 판정을 각각 동결한 뒤 case별16건 범주가 모두 같음을 확인했다. 새 비교 계약과 registration을 source값 개봉 전에 저장했고, 기존 sealed16 값만 한 번 대조했다. CSV/Gold/F2/예측/영상은 추가로 읽지 않았다.

- 명시적 synovitis 진단명2건: Dread/Qwen 모두양성. 직접 활막 형태 묘사는 없으므로 영상 정답 확인은 아니다.
- proxy-only12건: Dread양성6/Qwen양성6. 관절액·혈관절증·Hoffa 등의 표현을 임상 음성으로 바꾸지 않는다. 총수 같음은 같은 사례나 동일한 성능을 뜻하지 않는다.
- 미언급2건: 양source모두음성. 보고서 미언급이 영상 음성임을 입증하지 않는다.
- 명시적 음성0. 따라서 source false positive나 영상 오류율을 판정하지 못한다. 의도적source층화표본이므로 모집단 오류율 추정도 안 한다.

현 표본에는 직접 수정할 Dread의 명시적 보고서 누락/반대판정 근거가 부족하다. unknown/proxy의 loss 변경은 별도 실험 가설일 뿐 검증된 라벨 교정이 아니다. 자동mask/라벨교체/학습/표본확대를 하지 않는다. 추가 텍스트 감사를 자동 이어 붙이기보다 현재 개발 후보의 제출 가능성과 공정한 새group 대조를 구체화한다. Silver 전체가 정확하다거나 감독 개선이 불필요하다는 결론은 아니다.

[result.json](result.json) · [개봉 전 등록](registration.json) · [원래 비교 범위](../root-comparison-scope-v1.json). 원판정/원source 불변. 기존 sealed에는 Qwenlabel만 있어 mentioned/negated/Ppresent는 이번 대조에서 추정하거나 새로 읽지 않았다.
