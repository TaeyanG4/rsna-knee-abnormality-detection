# EXP086 이후 다음 한 투자 — 결정용 제안

2026-10-04 06:35 KST, dicom_header_io. **지금 권고하는 한 실험은 DINOv2-S의 CLS/window MIL을 마지막 공간 patch-token MIL로 바꾸는 단일8epoch 후보이다.** 원generic 초기값·영상·감독·optimizer를 유지하고, 완료된 EXP086 V2 CLS와 원 EXP084 B를 대조로 재사용한다. 이는 root의 CPU 준비/실행 결정 전 제안이며 구현·등록·학습을 하지 않았다. DINO를 계속 튜닝하기 위한 포괄 허가나 승산의 수치 예측이 아니다.

**06:36 root 결정:** 아래 producer의 patch-token MIL 제안은 대안으로 보존하고, 실제 다음 작업은 **추가 series의 가용 정보·고정K 계약을 확인하는 제한된 CPU feasibility**로 선택됐다. 큰 DINO 저하만으로 CLS 원인을 특정할 수 없다는 이유가 타당하다. 현재 문서는 proposal 기록이며 patch MIL 구현/학습 승인으로 사용하지 않는다. [root-next-decision.json](root-next-decision.json). 후속은 별도 배정 후 수행한다.

## 결과가 말하는 것과 아직 모르는 것

고정 보고서에서 B0.8933965 → DINO0.8288212, raw12macro Δ **−0.0645753**, paired component500 CI **[−0.0735314,−0.0539889]**. 8epoch/3448성공update/skip0, full635.438초, pilot31.641초, 원실패83.235초 포함750.314초이다. V1은결정론CUDA보간backward 실행FAIL이고 V2는그오류를해결한뒤완료한 **성능저하**다. 둘을섞거나 원FAIL을지우지않는다(M07).

12소견 모두점추정하락. ACL−.09582(P122), MCL−.14440(P45), Medial Meniscus−.10751(P358), Lateral Meniscus−.12550(P118), Baker−.10560(P110)가 macro손실 .048236, 전체차이의74.7%를차지한다. Fracture−.02529(P29)의 CI는0을포함하므로각소견모두통계적으로확실히나쁘다고말하지않는다. GE196/Philips271/Siemens378의macro도각각−.05774/−.06769/−.06278, all-nonzero654/some-zero234도−.06322/−.06819이다. 한scanner나mask구성하나로설명되는실패가아니다. 이산술은이미완료된report만사용했으며재평가가아니다.

동일softBCE train loss는 DINO1.0311→.8984, B1.0274→.8310으로내려갔다. 이는학습이진행됐다는관측이지만, DINO의LR가크다/작다·과적합·catastrophic forgetting 중무엇인지식별하지못한다. gradient/skip·초기값gate통과도좋은전이레시피의증거가아니다. **CoAt레시피를복사한generic DINO-S+CLS+8epoch**를현재B교체에채택하지않으며DINO전체·의료사전학습·긴학습전체를기각하지않는다(M05).

## 다섯 대안과 투자 순서

|순서·가설|구체적 비교·현재 근거|최소 비용과 한계·재검토 조건|
|---|---|---|
|**1. CLS로압축한공간정보가국소소견학습에불리하다**|현재code는ViT `global_pool='token'`, timm은`x[:,0]`만반환한다. 영상내256개contextual patch가소견별MIL에직접전달되지않는다. 마지막정규화patch를모두기존MIL에넘기는1candidate로이경로를검사한다. 아래정확대조참조.|새pixel/featurecache/모델다운로드0. CPU준비45–75분추정, GPU실측근거635초에head부하가추가된다. 후보11–20분은계획값·보장아님,32Tpilot≤180초/full≤1800초제안. 작은patch/전체해부학구조신호희석·수천token softmax·background shortcut위험. 한 번검사후이익없으면patch mean/max/grid를연속탐색하지않는다.|
|**2. 첫series/제한slot에서빠지는영상정보가병목이다**|원 B와동일K에서같은plane/FS의첫series+두번째비중복series를사전metadata규칙으로사용하고, train/eval모두반영하는단일coverage후보.1936/4407(43.93%)다중후보관측은가능범위이고개선대상비율이아니다. EXP081의고정가중치34건max-span교체와다른질문이다.|새pixelcache비용미측정. CPU설계1–3h와decode/I/O, B급full약43.5분기준후보1fit·보수cap60분이계획값. 원B입력분포를바꿔재학습하므로단순test-time교체반복금지. 더넓은상위권투자가설로보존하되지금patch1fit보다준비공백큼. patch실패후무조건이것을실행하는것도아님.|
|**3. encoder전용최적화/적응이필요하다**|random MIL head의초기gradient로표현이손상될가능성을묻는짧은head-only warmup→기존full fine-tune, 또는T-only MRI자기지도적응이후동일supervised8을비교할수있다. 현재loss하락·전target저하만으로어느방향인지모른다. LR범위검색대신먼저가설을고정해야한다.|head warmup후full은DINO단일fit수준+warmup이지만기간/해제시점근거가아직없다. SSL구현·CPU1–3h/GPU1–4h는post085거친예산을재사용한다. frozen현재계약probe가fine-tuned보다강하거나T-onlygradient/표현소실근거가생기면우선상향. 근거없는LR10배/추가epoch/layerwise계수grid는제안하지않는다.|
|**4. frozen현재입력linear probe로표현과학습문제를분리한다**|generic초기encoder의dense K24 CLS/patch-mean을같은G1에추출해고정train-only표준화+Ridge alpha100로study-levellinear진단가능. 필요하면최종fine-tunedencoder도같은feature/head계약에서비교해야‘fine-tuning 손실’해석이가능하다. 현재MIL output과linear-score차이를그렇게부르면안된다.|원EXP026 sparse18triplet/teacher평가결과 .78045는완료반대근거로재사용하며다시돌리지않는다. 현dense feature는새추출필요.4343×24×384 FP32약160.1MB/representation(둘320.2MB),평균후study특징은훨씬작다. encoder1개새추출계획5–15분+CPU head수분/준비30–60분,실측아님. 학습8epoch도10.6분이었던현재는별도probe가싸다고가정할수없어 **다음필수관문으로두지않는다**. 원frozen/최종2encoder비교는추출비용도2개다.|
|**5. 실제감독정의/미언급을고친다**|Dread/보고서의독립정의충돌이확인된target-cell만제한개입하고원감독대조·원reference평가를유지한다. 현재PFOA20/Synovitis16검토는끝났고자동교정근거부족. 새토론745576의불확실라벨제외무효는자기보고일뿐이다.|새전문가/독립영상근거의시간·비용미정,근거후B후보1fit약43.5분기준. Qwen confidence/불일치전체mask는근거없음. 같은라벨감사반복0. 원weak평가만으로임상정답개선의없음을결론내리지않는다.|

이순서는점수상승폭을추정한순위가아니다. **1번은공간집계라는명확한미검증개입을완료대조2개와낮은실행비용으로비교할수있어서** 우선한다. 0.0646큰격차가이한변경으로회복된다는보장은없다. 2·3·5번처럼큰입력/표현/감독투자를닫지않고,한번의cheap진단성학습이그투자를대체하며무한히이어지지않게한다(Biohub/M10).

## 권고 실험의 정확한 대조

**가설:** CLS는전체영상의contextual summary지만각patch는공간위치별contextual표현을남긴다. 표본수가작거나국소병변인소견에서소견별attention이patch를직접선택할수있게하면현재CLS/window경로보다나을수있다. 위손실집중은이가설을검사할이유이며CLS가원인이라는증거는아니다. patch token도transformer를통과해global context를포함하므로순수한국소texture만의대조라고하지않는다.

- 새candidate는 **원generic DINO asset/seed42/초기state에서처음시작**한다. 낮은성능의EXP086최종weight를추가학습하지않는다. V2 canonical518 trainable CPU-FP32 positional policy/strictFP16/내부1thread/worker12·평가2를재사용한다. 원asset/전체파라미터·MILhead초기값·원T3455/V888·known group·Dread원라벨·입력cache224·K12/K24·UID/epoch augmentation/window정규화는고정.
- 원`forward_features`의정규화된출력은 `[B*K,257,384]`이다. CLS prefix1개를제외한256patch를study안에서합쳐 **`[B,K*256,384]`를기존동일MIL head**에넘기는것만새개입으로한다. train3072/eval6144 token. 기존LayerNorm/attentionMLP256/Tanh/Dropout.2/12target softmax pooling/classifier파라미터를유지하고추가학습파라미터·새window·새crop·별도position/slothead는넣지않는다. padding/slot처리도원입력계약그대로다.
- **`timm.create_model(global_pool='avg')`로단순교체하지않는다.** 설치timm은pooltype에따라`norm`과`fc_norm`배치를바꾸므로동시에정규화위치/weight key까지바뀔수있다. 기존token설정의같은backbone을구성하고후단aggregation만명시적으로바꾸는안이다. 평균/최대/CLSconcatenate/중간layer혼합을동시에시도하지않는다.
- 동일AdamW backbone3e-5/head1e-3·wd.02/OneCycle.15·clip3·batch8/accum1·fixed8/3448성공update·skip0. 같은parameter 초기값/RNGseed를보존해도attention dropout의tensor크기가바뀌어global RNG소모는달라진다. pixel/shuffle UID stream은기존처럼독립이며,‘모든학습확률이동일한완전짝대조’라고주장하지않는다.
- **대조재사용:** 원EXP086 V2의완료CLS8예측과원EXP084 B8예측을그대로쓴다. 새CLS/원B 재학습0. 사전primary는candidate−**B raw12macro**로고정하고candidate−CLS를집계기메커니즘secondary로동시에계산한다. 낮은CLS만이겼다고성과개선을선언하지않는다. 기존두대조의UID/label/mask/input/source/runtime를새프로토콜에묶는다.
- 최종8epoch단한번의V예측만평가. 두contrast에같은paired component500draw·12소견P/N/CI·Baker/Fracture·scanner/mask group를계산한다. 단한seed·노출weakG1의조건부CI이며새독립검증아니다. Gold/중간V/순위혼합/targetrouting/소견선택없음.

## 최소 비용·실행 전 조건·중단 조건

**현재실행0.** root가선택하면CPU준비를최대75분의별도좁은범위로먼저결정하는안이다. 바뀐encoder출력shape/CLS제외/gradient/초기파라미터/RNG·기존head동일성·실제stream을검사하고, 원dataset/probe/성공serving전체검사를반복하지않는다. 실험source/output/receipt는새ID에둔다.

patch tensor를추가해도backbone forward횟수와pixels는같다. 추가attention중간값하나만계산하면batch8×K12×256patch×256hidden×FP16=12MiB이다. 이것은추가VRAM전체나안전한peak상한이아니며backprop/autocast/workspace를측정해야한다. 원V2 peak1.26GiB·epoch73초는출발근거이고새설정속도보장은아니다.

- T-only32update pilot1회/180초,초기4study유한추론,skip0/실제CPU보간과완료resource검사. full의 원3448step scheduler를 쓰며pilot결과로LR/batch/patch 수를바꾸지않는다.
- full 제안 cap1800초(30분),pilot+full1980초. `1.3*(last24전체update평균 초*3448)+300<=1800`을만족할때만root가full별도결정한다. 이전DINO635초를그대로ETA로쓰지않는다. ceiling은활동예산제안이며현재승인아니다. 실패/OOM/overflow/cap초과는기록후중단,자동retry/정밀도완화/patch subsampling없음.
- candidate가CLS보다높아도B보다여전히큰폭낮으면 **aggregation부분회복·B교체미달**로판정하고추가patchmean/LR/seed/fold/384를자동확대하지않는다. 두contrast모두0이하이면이번patch-MIL안미채택이지DINO/공간정보전체기각이아니다.
- B대비작은양의점추정·CI0포함은불확실후보로보존하며임의+.002폐기선은없다. B보다높아도Baker/Fracture나scanner편차를함께기록한뒤비용과다음독립확인에투자할지를다시결정한다. 공개LB를보고가중치/소견을다시맞추지않고이실험자체는제출후보가아니다.

## 재사용한 자료와 누락 점검

post085 brief의DINO/coverage/SSL/Silver현황·원asset·옛frozen반대근거를재사용했다. research-refresh-0603의DINOsaur는실제ConvNeXt-Tiny이며Gold선택·CSV family중복영향문제가있다. Plane-Aware252/depth변경은점수미검증이라성공처방으로채택하지않았다. 최신노트북/토론재수집·LB조회0.

M02: 현재V/공개teacher/Gold계보한계보존·새holdout주장없음. M05: 특정transfer/집계실패와encoder전체구분. M07:실행FAIL과완료저하구분. M10:기존대조재사용·한후보·실측기반예산·공개LB사후탐색없음. Biohub:손실기여/P/N/group로병목분해,작은효과보존,입력/표현/감독대안전체유지,공개기준본이나도구성공에고착하지않음. 세포tracking의분열/ILP수치·손익분기식을Knee에옮기지않았다.

근거파일의현재SHA·좁은읽기범위와소견산술은 [receipt.json](receipt.json)에저장한다. acquisition/training source는텍스트만읽었고import하지않았다. 모델load/weights·label/pixel/rawprediction접근/구현/등록/학습/GPU/다운로드/remote/중앙편집0. 현재root/다른worker가담당한작업을시작·중단하거나예약을변경하지않았다.
