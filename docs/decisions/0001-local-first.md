> 초기 설계 기록입니다. 2026-10-01 이후 개인용으로 범위를 조정했습니다. 현재 구현은 [실제 아키텍처](../architecture/implemented-dashboard.md), [추가 결정](../decisions/0002-personal-research.md), [STATE](../STATE.md)를 우선합니다. 팀 공유·아이디어 보드 등 아래 초기 제안은 현재 제품 기능이 아닙니다.

# ADR 0001 팀원별 독립 설치

상태: 채택. 근거: 2026-10-01 사용자 선택 '<owner remark removed>'.

브라우저 UI는 각 PC의 localhost 앱에 연결한다. SQLite/WAL·원문 파일·검색 인덱스·개인 설정·자격증명은 로컬에 둔다. Docker Compose를 재현 가능한 배포 경로로 제공하고 native 개발 경로도 지원한다. 학습 GPU와 대용량 데이터는 기존 환경에 둔다.

공통 코드와 설계 문서는 Git으로 버전을 맞추고, 실험·연구 기록은 origin과 revision/hash를 가진 export/import bundle로 교환한다. 같은 EXP 표시 이름을 전역 ID로 사용하지 않는다. DB 자동 양방향 sync와 공용 PostgreSQL은 MVP 범위 밖이다.

장점은 독립 커스텀, 개인 credential 유지, 오프라인 조회, 공용 서버 비용 제거다. 대가는 실시간 공유 상태 부재와 수집 중복이다. freshness를 보여주고 source bundle 재사용으로 줄인다. 공통 실험 계약을 바꿀 때는 version을 올리고 비교 가능성을 명시한다.

공용 서버를 기본으로 검토했던 초기 안은 이 선택으로 대체되었다. 향후 운영 방식을 바꿀 때 별도 ADR을 작성한다. 개인 실행 환경을 은밀히 중앙 서비스 의존 구조로 바꾸지 않는다.
