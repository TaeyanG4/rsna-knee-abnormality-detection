# 2026-10-02 영상 분석 진입 시 빈 화면 복구

사용자가 외부 브라우저에서 영상 분석으로 이동한 뒤 화면 전체가 빈 스크린샷을 제공했다. 현재 API·신규 브라우저에서는 정상, 이전 build chunk는404임을 확인했다. 사용자 외부 브라우저 콘솔에 직접 접근한 것은 아니므로 아래는 재현된 결함과 수리 근거다.

## 진단

- `/api/health`200, `/api/viewer/catalog`200(캐시2·raw4410), 현재 ViewerPage-PPrMdW2M.js200.
- 이전 ViewerPage-DQ2No9fC.js와 ViewerPage-DzRKeaph.js는404. Vite의 `emptyOutDir: true`가 새 build마다 이전 asset을 제거했다.
- 현재 `#data`를 열고 ViewerPage JS를404로 응답하도록 브라우저 테스트에서 주입한 후 영상 분석 버튼 클릭: body 텍스트가 빈 문자열이며 `Failed to fetch dynamically imported module` 발생. 동일 페이지 정상 요청은 오류 없이 표시됐다.
- React Suspense는 로딩 중 상태만 처리하며 실패한 import를 처리할 ErrorBoundary가 없어 root 전체가 제거됐다. 이전 검증은 새 탭 중심이어서 업데이트 전에 열린 탭을 놓쳤다.

## 수정

- `monitoring/web/vite.config.ts`: `emptyOutDir: false`. 기존 content-hash JS/CSS/font 보존, 새 index가 새 build를 가리킨다. 삭제된 과거 hash 파일을 다른 내용으로 위조하지 않았다.
- `monitoring/web/src/ScreenErrorBoundary.tsx`: 실패 안내와 명시적 새로고침. 자동 reload loop 없음. 저장하지 않은 메모가 초기화될 수 있음을 안내한다.
- `main.tsx`: 영상 분석·공식 안내·공식 문서의 각각의 오류 경계와 마지막 root 경계. 화면 로딩 오류 시 주변 메뉴를 유지한다.
- `style.css`: 오류 안내의 읽기 쉬운 글자·버튼.
- `monitoring/kneeops/api.py`: 루트HTML과API no-store, 정상 `/assets/*`는immutable 캐시.404는immutable이 아니다.
- 테스트: `monitoring/tests/test_static_assets.py`, `monitoring/web/tests/screen-recovery.spec.ts`.
- README, STATE, dashboard runbook 갱신.

## 실행·검증

```powershell
.venv/Scripts/python -X utf8 -m pytest monitoring/tests/test_static_assets.py monitoring/tests/test_core.py -q --basetemp=.local/test-viewer-recovery-01 --tb=short
# 11 passed
# monitoring/web에서:
npm run build
# TypeScript/Vite 성공
npx playwright test tests/screen-recovery.spec.ts tests/viewer.spec.ts tests/raw-data.spec.ts --reporter=line
# 최초 7 passed, 1 failed: reload 횟수 assertion이 hash 이동도 세는 테스트 문제
npx playwright test tests/screen-recovery.spec.ts --reporter=line
# main frame document request만 세도록 테스트 수정 후 3 passed
```

브라우저 복구3개와 기존viewer/raw5개 모두 통과. JS 실패와 CSS 실패 각각 메뉴 유지→새로고침→실제MRI 렌더링, 지속 실패 시 loop 없음과 raw query 보존을 확인했다. 테스트 fixture의404는 의도적이다.

실제 이전 index HTML을 `.local/ui-entry-before-viewer-fix.html`에 저장하고 새 build 후 이HTML로현재서버를열어 ViewerPage lazy load와 MRI 캔버스를 확인했다. 이전 entry `index-CKTBb9jz.js`, canvasVisible=true, pageerror없음. 구버전asset보존 동작이 실제로검증됐다.

사용자의 기존 Codex IAB 탭을 새로고침→영상 분석→첫study 선택까지 실행했다. AX에는 MRI캔버스·288×288·slice 표시가 나타났고 실제스크린샷에MRI가보였다. 외부브라우저의기존빈탭은사용자가Ctrl+Shift+R로새로고침해야한다. 원본/캐시 파일을 변경하거나 추가 다운로드하지 않았다.

서버 재시작 전에 다운로드·수집 active가 없는 것을API로확인했다. 정상stop/start후 백그라운드PID3128, run ID `1335e7dd68bd471a873d1ce2c3577e71`(관측값). 이미지 검토세션은 재시작에 초기화되지만 저장된DB·검토기록·다운로드는 보존된다.

## 제한·후속

- 이미삭제된과거파일을참조하는기존탭에는새오류경계코드도없다. 이번한번은새로고침이필요하다.
- asset은삭제하지않으므로빌드를반복하면정적파일이소량누적된다. 열려있는탭의버전을삭제하지않도록유지보수때만정리를검토한다.
- 브라우저네트워크실패나진짜render버그모두오류안내로표시한다. 반복된다면해당탭의콘솔오류를확인하며데이터손상이라고추정하지않는다.
