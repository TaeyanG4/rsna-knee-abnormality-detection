# 2026-10-03 17:01 KST · R2 두T4 study분할 측정 준비

- 상태: 준비/CPU검사완료,성능미측정. 기록자: Codex 하위작업. 새 `tools/d01_r2_shard_benchmark.py`·최소테스트·별도stage만 추가,기존도구/원본/훈련4/게이트/공동STATE·INDEX 수정 없음.
- 설계: 성공한runtime100 worker소스 bytes 그대로 재사용,고정UID짝/홀50+50을GPU0/1 child각CPU2thread로동시실행. 실제T4정확히2개·affinity/cgroup실효CPU≥4가아니면중단. legacy reader/5모델/FP16strict/gdcm고정,header최적화미적용.
- gate: merged5×100×12 float32/finite·UID중복누락/순서·모델identity/완료·원래volume/mask/fullreceipt/normalizedtensor exact. 기존수치max.002/mean.0005를새all5×100범위에사전고정,각member도판정. 실패시근거보존,기존input107/codec gate대체안함.
- 검증: CPU11검사통과(merge오류·max/mean·실제child timeout/형제실패시회수). POSIXkillpg는코드검토만,실제T4/분할효과미확인. notebook/payloadAST·SHA·과거worker원문·protocol동일검사완료. [증거](d01-r2-shard-stage.json).
- 실행준비: `workspace://.local/d01-r2-shard-staging/20261003-v3` 비공개GPU/offline701,264bytes. root가검토후 `.venv/Scripts/kaggle.exe kernels push -p "H:/kaggle/competitions/rsna-knee-abnormality-detection/.local/d01-r2-shard-staging/20261003-v3" --accelerator NvidiaTeslaT4 --timeout 1800`. 세션최대30분=2T4최대1GPUh,감독전체26분기한·실패/초과시두child그룹종료. cold로드·설치·병합포함wall/childVRAM/treeRSS기록. 계정쓰기·GPU실행없음.
- 다음: root가실측회수후입력/확률gate와공동9h여유판정. 물리I/O경합으로속도개선없을수있으며작은효과도비용과함께보존/보류한다. 유효성입증전serving기본값변경없음.
- 17:03추가검증: 기존worker.run은len(ids)에따라출력/완료를검사하므로50허용,thread2후속덮어쓰기없음. metadataGPU는bool true·데이터셋중복없음. 과거receipt를kernel version2에서정확재회수하여0a826…SHA동일확인,NPZ와같은version결합. stage의`root-review-verification.json`참조.
