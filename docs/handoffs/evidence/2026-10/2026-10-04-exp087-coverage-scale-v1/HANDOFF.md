# EXP087 scale CPU 구현 인계

- 10-04 08:09 KST, dicom_header_io. **구현 완료 / 실제 새 headers·pixels·GPU·model·labels0**. 중앙문서는 root 담당.
- 원 v3 helper와 성공8산출물 불변. 저장315(T252/V63) 그대로, 원8재디코딩0/남은307대안만 준비하는 별도 driver와 최소 B8 StudyWindows wrapper 구현.
- [상세](implementation-report.md), [SHA](final-bindings.json), 최종 [정확명령](freeze-v2/launch.json). v1은 미실행 초안으로 보존, v2가 최신. 315 source map 재열거 없이 결박만 보강.
- 새 header 계획15,774 / 대안 payload 상한3,046,923,720B / 최대3,684 planes. 제안900초·8GiB RSS·8GiB payload·65,536headers/512MiB parser. worker1/internal1, root 별도 실행결정 필요.
- focused synthetic15 PASS1.83초. 기존8 artifact 검사는 변경 없는 이전 PASS 재사용. K12/K24/원C/RNG/labels/다른window, batch4 collate, Windows pickle 및 source/cap gate 확인. v3 supervisor 본문 동일성 검사PASS.
- 새 output `.local/exp087-coverage/stage-v1` 아직 없음. root가 실제 source/fixture/plan을 reviewer에 전달→독립검토→CPU1회 여부 결정. 학습등록/모델/새cache실행/GPU는 이번 범위 밖.
- B8 genericinit/G1/8epoch/seed/recipe 및 batch4/K24 precision을 유지하면 원B 재사용. 입력변경만으로 control재학습하지 않음. whole315 실제tensor·GPU/eval 동등성/원307 DICOM replay는 미검증. weakG1·효능/임상/private/T4 미확정.
