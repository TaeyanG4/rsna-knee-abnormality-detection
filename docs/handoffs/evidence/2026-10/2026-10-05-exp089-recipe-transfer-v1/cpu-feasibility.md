# EXP089 · 단독 레시피 전이 비교 준비

두 완료 G1 가중치(224/8epoch, 384/12epoch)를 각각 사용한다. 학습군과 seed는 같고 실용 레시피 묶음은 다르다. 대회 영상 정답으로의 전이를 묻고, 해상도 단독 효과나 공개 기준본의 개선량을 측정한다고 주장하지 않는다.

공유 checkpoint loader와 입력 준비는 해상도를 지원한다. 두 해상도를 같은 multi-member 호출에 넣으면 계약 검사가 거부하므로 모델 한 개씩 별도 실행한다. 기존 inference entry의 OOF 경로는 Gold 제외 helper를 읽어 그대로 사용하지 않는다. 명시 UID만 받는 작은 wrapper가 필요하다.

기존 100건 DICOM/cache 일치와 decoder·offline wheel 근거는 재사용한다. 새로운 384 가중치·resize·batch1·T4 수치와 자원은 아직 미측정이다. 먼저 소수 V의 저장 확률을 정답 없이 비교하고, 검토된 새 source와 예산으로 engineering100/T4·공개3을 순차 판정한다. 완료 V888 AUC를 다시 계산하지 않는다.

이번 단계는 CPU source 읽기뿐이다. 모델/배열/라벨/OOF 로드·GPU/remote/제출·새 구현은 모두 0. 완성 실행 명령은 아직 없고 추정해 실행하지 않는다. 다음 구현·실행 계약을 별도 최소 검토한다. [가설·예산](../../../../../plan.md#exp089_recipe_transfer_g1--cpu-실행성-준비).
