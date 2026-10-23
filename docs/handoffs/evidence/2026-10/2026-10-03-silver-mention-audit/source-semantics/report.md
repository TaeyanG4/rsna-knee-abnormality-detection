# 2026-10-03 · Silver 필드 의미와 다음 가림 감사

**우선안은 PFOA의 중증도 불일치 감사, 가장 짧은 대안은 Baker’s의 크기 기준 감사, 세 번째는 Synovitis의 간접 소견 혼동 감사다.** 이 순서는 공식 정의와 저장된 생성 prompt의 차이로 정했다. 실제 오류율·모델 점수·EXP083 결과로 정한 순서가 아니다. 현재 학습 감독은 Dread soft labels이며, Qwen의 prompt 불일치가 Dread 라벨 오류를 증명하지는 않는다. Qwen 필드는 감사 후보를 만드는 보조 자료다.

## 확인한 출처와 한계

dataset11780375의 메타데이터는 slug의 Qwen3-8B와 달리 v2를 **Qwen2.5-32B**로 설명하며 저장된 generator notebook을 연결한다. 앱 Store의 현재 저장본은 notebook version10, revision `b4d46388…fda2d3`, 관측2026-10-02이다. 로컬 ipynb SHA는 `f04c0332…1c626`이며 Store 내용 SHA와 다른 객체다. prompt·parser·writer의 실제 source만 읽었고 출력·Gold 평가 cell16은 열지 않았다. **이 notebook revision이 보유 CSV를 실제 생성했다는 실행 receipt/정확한 dataset version은 미확인**이다. 최신 Kaggle 상태를 확인한 것도 아니다.

[출처 manifest](source-manifest.json), [생성 코드 발췌](generator-source-excerpts.txt), [공식 정의/주최 답변 발췌](official-definition-excerpts.txt)에 ID/revision/SHA/원 cell·줄 번호를 남겼다. 데이터 카드의 과거 Gold 성과 표기는 provenance 확인 중 보였으나 후보 순서나 규칙 선택에 사용하지 않았다. 원 raw/filled CSV·Report·라벨값·F2/Gold·예측은 읽지 않았다.

## 실제 필드 의미

|필드/상황|저장 source가 정의·수행하는 일|감사에서의 해석|
|---|---|---|
|`mentioned`|cell6 L21–23: finding뿐 아니라 관련 구조를 명시적으로 논의하면1. 무정보면0|병명이 없다는 것과 구조/소견 미언급은 다르다. 의학적 음성 표시가 아니다|
|`negated`|cell6 L25–28: absent/normal/intact/preserved 또는 relevant pathology 부정|명시적 부정의 추출 결과다. 진단 영상 음성의 증명이나 중증도 미달과 동일하지 않다|
|`confidence`|cell6 L30–40·cell8 L50–53: `probability`를 같은 CSV 이름에 저장하는 모델의 P(present)|텍스트 판독 신뢰도나 검증된 보정 확률이 아니다. 그대로 loss 신뢰도 가중치로 쓰면 안 된다. v1 predicted-class confidence와 혼합 금지|
|미언급|cell6 L44–45와 cell8 L73–74: label0/mentioned0/negated0. probability는 별도로 유지|`label=0`만으로 확정 음성이라고 해석하면 안 된다. 미언급 probability의 통일된 의미도 보장되지 않는다|
|불확실/중증도 경계|cell6 L36·48–50: possible/cannot exclude/borderline는 확률을 낮추고 서술 근거로 분류|별도 `unknown`/`uncertain` 필드가 없다. 중간 확률만으로 불확실 문장·중증도 미달·실제 중간 위험을 분리할 수 없다|
|누락 target/field|cell8 L55–78: 찾지 못한 target은 label0/mentioned0/negated0/probability0.5 기본값. cell14 L16–32는 parse 성공 시 status=ok|`status=ok`와 전체12target 의미적 완결성은 다르다. 이 기본값은 진짜 미언급과 구분되지 않을 수 있다|
|generation error|cell14 L100–109: status=error이며 모든 target0/0.5/0/0|오류·파서 누락·실제 미언급을 합치지 않는다. 전체 raw coverage는 병렬 담당이 확인한다|
|filled 파일|cell14 L5–12·126–143: unmentioned target의 confidence만 OA/effusion partner에 따라 상향|원 label/mentioned/negated는 바뀌지 않는다. raw와 filled는 교환 가능한 감독이 아니며, 이번 감사는 raw source 의미만 확인했다|

parser는 명시부정이면 label0으로 고치지만 probability를 낮추지는 않는다. binary 필드 범위·모든 target 존재·확률과 label의 의미적 일관성을 완전히 검증하지도 않는다. `_match_key`의 substring fallback(cell8 L29–43) 및 기본값은 확인 대상이지, 보유 CSV에 실제 오류가 있다는 관측은 아니다. 저장된 `qwen_raw`는 최대4000자(cell14 L24/L102)이므로 전체 원출력 복원이 항상 가능하다고 가정하지 않는다. `generation_seconds`도 이 source의 `process()` 안에서는 생성 후 parse 시간을 재므로 LLM 전체 추론시간으로 쓰지 않는다.

## 오류 가능성이 높은 세 가지 감사안

|우선안|오류 가능 메커니즘과 직접 근거|텍스트로 확인 가능한 경계 / 영상이 필요한 경계|다음 작은 train 가림 감사 제안|예상 사람 검토 시간|
|---|---|---|---|---|
|**1 PFOA**|공식733343 L65–69는 약1cm 이상 범위·두께50% 초과 연골 소실. prompt cell6 L91–95는 연골연화/연골 소실·골극·연골하 변화만으로 양성 허용. 기준이 명백히 더 넓다|명시된 low-grade/얕은/작은 연골 손상은 불일치 후보. 부위·범위·깊이 없는 chondrosis는 **정보 부족**, 자동 음성이 아니다. 실제 범위·두께는 영상 판독 필요|최대80건: 무조건부 무작위20 + 명시정상/부정, 명시 저등급, 불특정·불확실 연골 변화, 명시 고등급의 각 최대15. 중복 제거·희소 strata는 있는 만큼만. 부위/등급/범위/근거 문장 기록|90–150분 계획 추정; 전문의 영상판독 별도|
|**2 Baker’s**|공식 L75는 중등도/대량의 특징적 위치 액체 저류. prompt L139–144는 명시된 cyst면 양성이고 작은 정상 bursa fluid만 제외. 작은 실제 cyst의 기준 차이가 남는다|작은/미세한 cyst·명시 부정·중등도/대량은 텍스트 층화 가능. 단순 “Baker cyst”는 크기 정보 부족. 파열·감압된 cyst를 작은 현재 크기만으로 음성 확정하지 않는다|최대60건: 무조건부 무작위20 + 명시 작은 cyst/정상 bursa, 중등도·대량 cyst, 크기 미상/불확실, 명시 부정의 각 최대10. cyst/bursa/다른 위치 낭종 구분|45–90분 계획 추정; 세 안 중 가장 짧은 경계 확인|
|**3 Synovitis**|공식 L73는 활막의 염증·비후. prompt L117–137는 Hoffa 부종/충돌, 염증성 plica 등 간접 표현을 폭넓게 포함. bare effusion은 제외한다고 명시|명시 활막 염증·비후, Hoffa만, plica만, fluid만, 부정을 텍스트에서 구분할 수 있다. 간접 소견만으로 공식 활막 병변을 양성 또는 음성 확정할 수 없어 전문 영상 판단 의존도가 높다|최대80건: 무조건부 무작위20 + 명시 활막 병변, Hoffa/plica 간접 소견, effusion-only, 부정/무언급·불확실의 각 최대15. 대리 소견과 실제 활막 기술을 별도 기록|120–180분 계획 추정; 세 안 중 임상 판정 의존도가 큼|

위 시간은 새 가림 텍스트 검토의 예상 인력시간이며 실측·전문의 견적·GPU 예산이 아니다. 준비/표본 동결20–40분은 별도이며, 원문 언어·서술 길이에 따라 달라진다. 세 안을 모두 실행하는 제안이 아니다. 병렬 coverage 결과와 source 연결을 확인한 뒤 **하나만** 선택한다. 공식 MCL 고등급 급성파열과 prompt의 일반 sprain/비후 양성도 명백한 감사 후보지만, 기존 MCL 감사와 중복 확대하지 않고 별도 가설로 보존한다. Effusion도 중등도/대량 기준이 명시되어 있으나 이번 세 안의 자동 추가 대상은 아니다.

## 가림과 판정 계약

후속 허가가 있을 때만 원3475 train UID와 known component 안에서 표본을 고정한다. `SHA256(고정salt,component)` 등의 결정적 선택을 사용하고, random panel과 의도적으로 늘린 경계 strata를 구분해 원표본 inclusion 수를 보존한다. 희소한 strata를 찾으려고 전체 raw를 재감사하거나 목표건수까지 자의적으로 보충하지 않는다. 단순 strata 집계는 모집단 오류율 추정치가 아니다.

검토자에게는 가림 ID와 Report만 보여주고 Qwen/Dread 값·모델예측·상호 불일치·scanner·LB를 숨긴다. 사전 고정 rubric은 **명시적 공식 양성 근거 / 명시적 부정 / 명시적 중증도 미달 / 불확실 / 미언급 / 세부정보 부족**을 구분한다. 인용 문장·부위·정도·불확실 표현을 기록하고 중복 분류 우선순위를 사전 고정한다. 동결한 뒤에만 저장 weak 필드와 대조한다. 파서의 정상 status나 confidence0.5는 임상 rubric을 대신하지 않는다.

공식 은 영상 판독 원칙이지 **보고서 미언급=음성** 규칙이 아니다. 명확하게 기술된 사례만 mask로 남기면 중증도·기관별 보고 양식·검사 목적이 다른 집단을 선택해 편향을 키울 수 있다. 향후 학습 mask 실험은 규칙을 train에서 먼저 고정하고, 두 arm의 **평가집합·평가 mask는 공통**으로 유지한다. 전체 train 대비 포함률·제외 사유·장비/언어/소견 구성 변화를 함께 확인하며, 선택된 subset의 weak 점수 향상을 전체 임상/private 효과로 바꾸지 않는다.

주최733826의 답변은 영상 라벨이 독립 판독이고, 보고서와 다를 때 영상이 우선이며, 공개 음성도 실제로 부재로 주석했다고 명시한다. 따라서 텍스트 감사는 잘못된 추출과 정의 차이를 찾아내는 단계다. 새 영상 정답 생성·자동 relabel·unknown 전체 masking·Gold 재검증을 수행한 것이 아니다.
