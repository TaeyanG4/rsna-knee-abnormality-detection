# 구현·데이터 참고 현황

실제 진행과 결과는 [HANDOFF](../HANDOFF.md), 다음 계획은 [plan](../plan.md), 교훈과 실패는 [discoveries](../discoveries.md), 공통 지시는 [AGENTS](../AGENTS.md).

- 개인 KneeOps localhost:8765·PC별 계정/runtime DB. [범위](decisions/0002-personal-research.md) · [운영](runbooks/dashboard.md).
- Layout v2·knee_workspace/config/workspace.toml. [경로](runbooks/workspace-layout.md).
- 자료·실험·제출 검색/비교와 MRI/ROI/EDA/전처리 비교 기능. [구현](architecture/implemented-dashboard.md) · [영상](runbooks/image-viewer.md).
- ZIP 265GB·train 4407/test 3. 전수 감사 미완료. [데이터](runbooks/training-data.md).
- 보고서 그룹 fold와 새 그룹 계약. Gold58 선택 노출·Silver unknown/confidence loss 미적용·test Report 없음. 해석은 discoveries M02/M03/M06. [학습](runbooks/training-raptor.md).
- 앱/Kaggle .venv와 GPU Python 분리. 학습/서빙·GPU 큐·원격 bundle·receipt 도구. [도구](runbooks/tools.md) · [원격](runbooks/remote-gpu.md).
- 같은 작업 공간에서 새 Codex/Claude CLI 세션을 호출하고 결과를 회수하는 도구. [사용법과 제한](runbooks/tools.md#로컬-에이전트-세션).

개별 실험 결과는 여기서 중복 갱신하지 않는다. [이전 현황 원문/hash](handoffs/archive/snapshots/2026-10-04-agent-docs-v1/manifest.json).
