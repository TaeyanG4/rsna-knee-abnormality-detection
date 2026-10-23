# EXP087 정책 집계 독립검토

- 완료·`/root/precision_code_review`·2026-10-04 07:39 KST. **집계·정책 일치 PASS, hash 표기 주의1건.** 세부 검사·SHA: [review](independent-review.json).
- 저장 decision/source/result/detail과 passive sampler만 읽었다. 원metadata·mask·census·픽셀·header·label·OOF·model·GPU 접근/재실행0. 중앙문서/원본 변경0.

252/3455=7.2938%(T),63/888=7.0946%(V). 사유 합계는 각각3455/888이며, 저장된315개 detail은 중복UID 없이 두 split로 나뉜다. detail로 모든 epoch별 active/window 수와 요약을 재합산해 정확히 일치했다.

소스는 original/alternate Coronal·fluid1·같은 알려진FS, 기존5slot에서 미사용, lexical첫후보를 적용한다. SLOTS64의 slot2는32..43, 유효 triplet centre33..42다. `count//2`는 **0-based 홀수(두번째·네번째…) 발생 교체**와 일치한다. 1-based 홀수 의미로 구현하면 다른 정책이므로 indexing을 명시해야 한다.

T8epoch 잠재window1742/(3455×8×12)=0.5252%, V126/(888×24)=0.5912%. T는 drop_last 전의 명목 분모/기회이며 실제 학습노출률이 아니다. V의 train_*는 가상RNG 수치·학습미사용이고, T의 eval_*도 평가 sampler 진단이다. alternate geometry/SOP·영상중복·mask를 확인하면 같은 고정정책의 허용교체 수는 줄 수 있다. 정보이득·AUC효과·환자독립성의 근거는 아니다.

주의: `potential_detail_sha256`는 raw 파일hash와 다르다. 원코드가 LF payload를 hash한 뒤 Windows `write_text`로 CRLF 저장했다. 5,672개 newline 변환만 되돌리면 저장hash와 정확히 일치하여 내용 불일치는 없다. 원파일을 보존하고 LF-payload hash/실제 raw hash를 별도 receipt로 명명하면 된다. 재집계 불필요.

M03/05/09/10 적용. 원입력은 bytes/mtime만 기록돼 이 검토가 원metadata 불변성까지 인증하지 않는다. producer의 exactG1 ID검사는 코드에서 확인했으며 원UID census를 반복하지 않았다. 현재 sampler SHA를 남겼지만 실행시 helper SHA의 소급증명은 아니다. 남은 일: root의 정책 채택 판단과 hash 표기 연결; 학습/전체cache 승인으로 해석하지 않는다.
