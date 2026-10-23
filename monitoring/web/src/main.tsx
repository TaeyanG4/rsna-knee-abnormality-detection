import React, { lazy, Suspense, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  BookOpen,
  Check,
  ChevronRight,
  Clock3,
  Database,
  ExternalLink,
  FlaskConical,
  GitCompareArrows,
  LayoutDashboard,
  LayoutGrid,
  List,
  Lightbulb,
  LoaderCircle,
  Menu,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Target,
  Trophy,
  Upload,
  X,
} from "lucide-react";
import "./style.css";
import DataPage from "./DataPage";
import ScreenErrorBoundary from "./ScreenErrorBoundary";
import SyncControl from "./SyncControl";
import SourceInsights, { MetaChips } from "./SourceInsights";
import DiffView from "./DiffView";
import "./analysis.css";
import "./official.css";
const OfficialPage = lazy(() => import("./OfficialPage"));
const ViewerPage = lazy(() => import("./ViewerPage"));
const PredictionsPage = lazy(() => import("./PredictionsPage"));
const OfficialDocument = lazy(() =>
  import("./OfficialPage").then((module) => ({
    default: module.OfficialDocument,
  })),
);

type RecordData = Record<string, any>;
type Page =
  | "predictions"
  | "data"
  | "viewer"
  | "overview"
  | "history"
  | "research"
  | "experiments"
  | "submissions"
  | "settings"
  | "official"
  | "notebooks"
  | "discussions"
  | "reports"
  | "own"
  | "solutions";
const pages: { id: Page; label: string; icon: typeof Activity; sub: string }[] =
  [
    {
      id: "overview",
      label: "대회 현황",
      icon: LayoutDashboard,
      sub: "오늘의 연구 방향을 정리하세요.",
    },
    {
      id: "research",
      label: "전체 자료",
      icon: BookOpen,
      sub: "수집한 자료를 검색하고, 원문과 버전 변화를 확인하세요.",
    },
    {
      id: "official",
      label: "공식 안내",
      icon: ShieldCheck,
      sub: "대회 규칙, 평가와 일정의 원문을 확인하세요.",
    },
    {
      id: "notebooks",
      label: "공개 노트북",
      icon: BookOpen,
      sub: "모델 코드와 버전별 변경 내용을 살펴보세요.",
    },
    {
      id: "discussions",
      label: "토론",
      icon: Lightbulb,
      sub: "대회 논점과 호스트 답변을 모아보세요.",
    },
    {
      id: "reports",
      label: "로컬 문서",
      icon: Database,
      sub: "이 작업 공간의 연구 문서와 보고서를 찾아보세요.",
    },
    {
      id: "own",
      label: "내 노트북",
      icon: BookOpen,
      sub: "로컬에 저장된 내 노트북을 확인하세요.",
    },
    {
      id: "solutions",
      label: "과거 솔루션",
      icon: Trophy,
      sub: "이전 대회의 참가자 솔루션과 원문 전략을 살펴보세요.",
    },
    {
      id: "history",
      label: "과거 대회 연구",
      icon: Lightbulb,
      sub: "유사 대회의 솔루션과 코드를 현재 문제에 연결하세요.",
    },
    {
      id: "data",
      label: "다운로드·캐시",
      icon: ArrowDownToLine,
      sub: "기존 캐시를 연결하고, 필요한 원본만 내려받으세요.",
    },
    {
      id: "viewer",
      label: "영상 분석",
      icon: Activity,
      sub: "MRI와 캐시를 직접 살펴보고 영상 품질·라벨을 점검하세요.",
    },
    {
      id: "experiments",
      label: "실험 기록",
      icon: FlaskConical,
      sub: "같은 검증 조건에서 의미 있는 개선을 찾으세요.",
    },
    {
      id: "predictions",
      label: "예측 비교",
      icon: BarChart3,
      sub: "로컬 예측 파일의 소견별 AUC, 모델 간 순위 상관, 동일 가중 blend를 비교하세요.",
    },
    {
      id: "submissions",
      label: "제출 이력",
      icon: Trophy,
      sub: "공식 평가 결과와 다음 제출을 점검하세요.",
    },
    {
      id: "settings",
      label: "내 작업 환경",
      icon: Settings2,
      sub: "이 PC의 화면과 자료 수집 방식을 설정하세요.",
    },
  ];
const sourceKinds: Record<string, string> = {
  official: "official",
  notebooks: "notebook",
  discussions: "discussion",
  reports: "report",
  own: "own_notebook",
  solutions: "solution",
};
const navGroups: { label: string; icon: typeof Activity; tabs: Page[] }[] = [
  { label: "대회 안내", icon: LayoutDashboard, tabs: ["overview", "official"] },
  {
    label: "자료 라이브러리",
    icon: BookOpen,
    tabs: ["research", "notebooks", "discussions", "own", "reports"],
  },
  { label: "과거 대회", icon: Trophy, tabs: ["history", "solutions"] },
  { label: "학습 데이터", icon: ArrowDownToLine, tabs: ["data", "viewer"] },
  {
    label: "실험·제출",
    icon: FlaskConical,
    tabs: ["experiments", "predictions", "submissions"],
  },
  { label: "내 작업 환경", icon: Settings2, tabs: ["settings"] },
];
const kinds: Record<string, string> = {
  solution: "솔루션 토론",
  leaderboard: "리더보드",
  official: "공식 안내",
  notebook: "공개 노트북",
  discussion: "토론",
  report: "로컬 문서",
  own_notebook: "내 노트북",
};
const CURRENT = "rsna-knee-abnormality-detection";
const shortDate = (value?: string | null) =>
  value
    ? new Date(
        value.endsWith("Z") || value.includes("+") ? value : value + "Z",
      ).toLocaleDateString("ko-KR", {
        year: "2-digit",
        month: "numeric",
        day: "numeric",
        timeZone: "Asia/Seoul",
      })
    : "—";
const metrics: Record<string, string> = {
  weak_oof: "Weak OOF",
  pseudo_oof_leaky: "노출된 pseudo OOF",
  diagnostic: "진단 지표",
};

const connectorNames: Record<string, string> = {
  notebook_versions: "노트북 버전 갱신",
  saved_evidence: "저장된 조사 자료",
  local_import: "로컬 자료",
  submissions: "내 제출",
  leaderboard: "리더보드 상위 100",
  leaderboard_full: "전체 리더보드·내 순위",
  official_pages: "공식 안내",
  notebook_discovery: "노트북 목록",
  notebook_sources: "노트북 소스",
  discussions: "토론",
};
let token = "";
async function api(path: string, options: RequestInit = {}) {
  const response = await fetch("/api" + path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-Kneeops-Token": token,
      ...options.headers,
    },
  });
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    const detail = error.detail ?? error.error ?? `요청 실패 (${response.status})`;
    throw Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json();
}
const date = (value?: string) =>
  value
    ? new Date(value).toLocaleString("ko-KR", {
        month: "numeric",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "Asia/Seoul",
      })
    : "아직 없음";
const score = (value?: number | null) =>
  value == null ? "—" : value.toFixed(3);
const addedDate = (value?: string) =>
  value
    ? new Date(value).toLocaleString("ko-KR", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "Asia/Seoul",
      })
    : "기록 없음";
const isToday = (value?: string) =>
  !!value &&
  new Date(value).toLocaleDateString("en-CA", { timeZone: "Asia/Seoul" }) ===
    new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Seoul" });
const safeURL = (url: string) => (/^https?:\/\//i.test(url) ? url : "#");
function Badge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return <span className={"badge " + tone}>{children}</span>;
}
function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="empty">
      <Database size={28} />
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}
function App() {
  const [page, setPage] = useState<Page>(
    (location.hash.slice(1) as Page) || "overview",
  );
  const [brief, setBrief] = useState<RecordData | null>(null),
    [prefs, setPrefs] = useState<RecordData>({
      theme: "light",
      compact: false,
      watch_targets: [],
      hidden_cards: [],
      sync_enabled: false,
      sync_minutes: 60,
    });
  const [session, setSession] = useState<RecordData>({ targets: [] }),
    [runs, setRuns] = useState<RecordData[]>([]),
    [jobs, setJobs] = useState<RecordData[]>([]);
  const [sources, setSources] = useState<RecordData>({ items: [], total: 0 }),
    [q, setQ] = useState(""),
    [kind, setKind] = useState(sourceKinds[location.hash.slice(1)] || ""),
    [offset, setOffset] = useState(0),
    [sort, setSort] = useState("added"),
    [bodyFilter, setBodyFilter] = useState(""),
    [inputRef, setInputRef] = useState(""),
    [topInputs, setTopInputs] = useState<RecordData | null>(null);
  const [runQ, setRunQ] = useState(""),
    [runKind, setRunKind] = useState(""),
    [runScope, setRunScope] = useState(""),
    [runSort, setRunSort] = useState("name");
  const [selected, setSelected] = useState<RecordData | null>(null),
    [diff, setDiff] = useState(""),
    [chosen, setChosen] = useState<string[]>([]),
    [comparison, setComparison] = useState<RecordData | null>(null);
  const [toast, setToast] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [menu, setMenu] = useState(false);
  const [competitions, setCompetitions] = useState<RecordData[]>([]),
    [scope, setScope] = useState(""),
    [newSlug, setNewSlug] = useState(""),
    [newTitle, setNewTitle] = useState(""),
    [versionNumber, setVersionNumber] = useState("");
  const sourceRequest = useRef(0);
  const sourceDialog = useRef<HTMLDialogElement>(null);
  const notice = (message: string) => {
    setToast(message);
    setTimeout(() => setToast(""), 5000);
  };
  const guard = async (task: () => Promise<any>) => {
    setError("");
    try {
      return await task();
    } catch (e) {
      setError((e as Error).message);
      return undefined;
    }
  };
  const refresh = async () => {
    const [b, r, j, c] = await Promise.all([
      api("/briefing"),
      api("/runs"),
      api("/jobs"),
      api("/competitions"),
    ]);
    setBrief(b);
    setRuns(r);
    setJobs(j);
    setCompetitions(c);
  };
  const searchSources = async () => {
    const request = ++sourceRequest.current;
    const data = await api(
      `/sources?q=${encodeURIComponent(q)}&kind=${kind}&offset=${offset}&competition=${encodeURIComponent(scope)}&sort=${sort}&body=${bodyFilter}&input=${encodeURIComponent(inputRef)}`,
    );
    if (request === sourceRequest.current) setSources(data);
  };
  useEffect(() => {
    guard(async () => {
      const s = await api("/session");
      token = s.token;
      setSession(s);
      setPrefs({
        research_view: "list",
        sync_history: [],
        ...(await api("/preferences")),
      });
      await refresh();
    });
  }, []);
  useEffect(() => {
    const timeout = setTimeout(() => guard(searchSources), 180);
    return () => clearTimeout(timeout);
  }, [q, kind, offset, scope, sort, bodyFilter, inputRef]);
  useEffect(() => {
    document.documentElement.dataset.theme = prefs.theme;
  }, [prefs.theme]);
  useEffect(() => {
    const handle = () =>
      setPage(
        pages.some((p) => p.id === location.hash.slice(1))
          ? (location.hash.slice(1) as Page)
          : "overview",
      );
    window.addEventListener("hashchange", handle);
    return () => window.removeEventListener("hashchange", handle);
  }, []);
  useEffect(() => {
    if (selected && !sourceDialog.current?.open)
      sourceDialog.current?.showModal();
    if (!selected) sourceDialog.current?.close();
  }, [selected]);
  useEffect(() => {
    const timer = setInterval(
      () =>
        guard(async () => {
          const next = await api("/jobs");
          setJobs(next);
          if (
            jobs.some((j) => ["queued", "running"].includes(j.status)) &&
            !next.some((j: RecordData) =>
              ["queued", "running"].includes(j.status),
            )
          ) {
            await refresh();
            await searchSources();
            notice("수집 작업이 끝났습니다. 아래 상태에서 결과를 확인하세요.");
          }
        }),
      4000,
    );
    return () => clearInterval(timer);
  }, [jobs, q, kind, offset, scope, sort, bodyFilter, inputRef]);
  const nav = (next: Page) => {
    location.hash = next;
    setPage(next);
    setMenu(false);
  };
  const isLibrary =
    page === "research" || (page in sourceKinds && page !== "official");
  useEffect(() => {
    if (page in sourceKinds) {
      setKind(sourceKinds[page]);
      setOffset(0);
    } else if (page === "research") {
      setKind("");
    }
    // Notebook and discussion tabs open on the current competition, newest activity first.
    if (page === "notebooks" || page === "discussions") {
      setScope(CURRENT);
      setSort("activity");
    } else if (page in sourceKinds || page === "research") setSort("added");
    setBodyFilter("");
    if (page !== "notebooks") setInputRef("");
  }, [page]);
  useEffect(() => {
    if (page !== "notebooks") return;
    api("/notebook-inputs?limit=24&competition=" + encodeURIComponent(scope))
      .then(setTopInputs)
      .catch(() => setTopInputs(null));
  }, [page, scope]);
  const shownRuns = runs
    .filter(
      (r) =>
        (!runQ ||
          (r.name + " " + r.family)
            .toLowerCase()
            .includes(runQ.toLowerCase())) &&
        (!runKind || r.kind === runKind) &&
        (!runScope ||
          (runScope === "12"
            ? r.targets.length === 12
            : runScope === "partial"
              ? r.targets.length < 12
              : r.score != null)),
    )
    .sort((a, b) =>
      runSort === "score"
        ? (b.score ?? -1) - (a.score ?? -1)
        : runSort === "observed"
          ? String(b.observed_at).localeCompare(String(a.observed_at))
          : String(b.name).localeCompare(String(a.name)),
    );
  const active = pages.find((p) => p.id === page) || pages[0];
  const group = navGroups.find((g) => g.tabs.includes(page)) || navGroups[0];
  const working = jobs.some((j) => ["running", "queued"].includes(j.status));
  const jobsKey = jobs.map((j) => j.id + ":" + j.status).join(",");
  const scheduleChanged = (enabled: boolean, minutes: number) =>
    setPrefs((p) => ({ ...p, sync_enabled: enabled, sync_minutes: minutes }));
  const start = async (jobKind: string, ref = "") =>
    guard(async () => {
      setBusy(true);
      try {
        await api("/jobs", {
          method: "POST",
          body: JSON.stringify({ kind: jobKind, ref }),
        });
        setJobs(await api("/jobs"));
        notice("수집을 시작했습니다. 화면을 계속 사용할 수 있습니다.");
      } finally {
        setBusy(false);
      }
    });
  const openSource = async (id: string, revision?: string) =>
    guard(async () => {
      setSelected(
        await api(
          "/source?id=" +
            encodeURIComponent(id) +
            (revision ? "&revision=" + revision : ""),
        ),
      );
      setDiff("");
      setVersionNumber("");
    });
  const savePreferences = () =>
    guard(async () => {
      setPrefs(
        await api("/preferences", {
          method: "PUT",
          body: JSON.stringify(prefs),
        }),
      );
      notice("이 PC의 설정을 저장했습니다.");
    });
  const days = brief?.competition.deadline
    ? Math.max(
        0,
        Math.ceil(
          (new Date(brief.competition.deadline).getTime() - Date.now()) /
            86400000,
        ),
      )
    : null;
  return (
    <div className={"app " + (prefs.compact ? "compact" : "")}>
      <aside className={"sidebar " + (menu ? "open" : "")}>
        <a className="brand" href="#overview">
          <span className="brandmark">
            <Activity size={23} />
          </span>
          KneeOps<span className="branddot">●</span>
        </a>
        <div className="workspace-label">COMPETITION WORKSPACE</div>
        <div className="competition-select">
          <span className="rsna-mark">R</span>
          <div>
            <strong>RSNA Knee</strong>
            <small>Abnormality Detection · 2026</small>
          </div>
          <ChevronRight size={15} />
        </div>
        <nav aria-label="주 메뉴">
          {navGroups.map((g) => (
            <button
              key={g.label}
              className={group === g ? "nav-item active" : "nav-item"}
              aria-current={group === g ? "page" : undefined}
              onClick={() => {
                setQ("");
                setScope("");
                setOffset(0);
                if (g.tabs[0] === "research") setKind("");
                nav(g.tabs[0]);
              }}
            >
              <g.icon size={20} />
              <span>{g.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <div className="status-dot" />
          <strong>내 PC 연구실</strong>
          <p>
            설정과 수집 자료는 이 PC에 저장됩니다.
            <br />
            외부 서버 없이 개인용으로 사용하세요.
          </p>
        </div>
        <div className="sidebar-footer">
          <span className="avatar">ME</span>
          <div>
            <strong>개인 워크스페이스</strong>
            <small>
              {session.instance_id?.slice(0, 8) || "연결 중"} · Local
            </small>
          </div>
          <ShieldCheck size={17} />
        </div>
      </aside>
      <main>
        <header className="topbar">
          <button
            className="icon-button mobile"
            onClick={() => setMenu(!menu)}
            aria-label="메뉴 열기"
            aria-expanded={menu}
          >
            <Menu size={22} />
          </button>
          <div className="breadcrumb">
            연구실 <ChevronRight size={13} />
            <strong>{group.label}</strong>
          </div>
          <div className="top-right">
            <span className="local-pill">
              <span className="status-dot" />
              로컬 연결
            </span>
            <a
              href="https://www.kaggle.com/competitions/rsna-knee-abnormality-detection"
              target="_blank"
              rel="noreferrer"
            >
              Kaggle에서 보기 <ArrowUpRight size={14} />
            </a>
          </div>
        </header>
        <div className="content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">RSNA KNEE ABNORMALITY DETECTION</div>
              <h1>{group.label}</h1>
              <p>{active.sub}</p>
            </div>
            {page !== "data" && page !== "predictions" && (
              <div className="heading-actions">
                <button
                  className="button secondary"
                  onClick={() => start("import")}
                  disabled={working || busy}
                >
                  <Database size={16} />
                  <span className="button-label">로컬 자료 읽기</span>
                </button>
                <button
                  className="button primary"
                  onClick={() => start("sync")}
                  disabled={working || busy}
                >
                  <RefreshCw size={16} className={working ? "spin" : ""} />
                  <span className="button-label">
                    {working ? "수집 중" : "Kaggle 새로고침"}
                  </span>
                </button>
              </div>
            )}
          </div>
          {group.tabs.length > 1 && (
            <nav className="section-tabs" aria-label={group.label + " 분류"}>
              {group.tabs.map((id) => {
                const item = pages.find((p) => p.id === id)!;
                return (
                  <button
                    key={id}
                    aria-current={page === id ? "page" : undefined}
                    className={page === id ? "active" : ""}
                    onClick={() => {
                      setQ("");
                      setOffset(0);
                      nav(id);
                    }}
                  >
                    {id === "official" ? "개요·규칙" : item.label}
                    {id in sourceKinds && id !== "official" && (
                      <span>
                        {(["notebooks", "discussions", "own"].includes(id)
                          ? brief?.competition_source_counts
                          : brief?.source_counts
                        )?.[sourceKinds[id]] || 0}
                      </span>
                    )}
                  </button>
                );
              })}
            </nav>
          )}
          {error && (
            <div role="alert" className="alert error">
              <span>{error}</span>
              <button
                className="icon-button"
                onClick={() => setError("")}
                aria-label="오류 닫기"
              >
                <X size={17} />
              </button>
            </div>
          )}
          {working && (
            <div className="alert">
              <LoaderCircle className="spin" size={17} />{" "}
              {
                jobs.find((j) => ["running", "queued"].includes(j.status))
                  ?.progress
              }{" "}
              · 원본 자료를 확인하고 있습니다.
            </div>
          )}
          {!brief ? (
            <div className="loading">
              <LoaderCircle className="spin" />
              연구실을 여는 중…
            </div>
          ) : (
            <>
              {page === "data" && <DataPage api={api} />}
              {page === "viewer" && (
                <ScreenErrorBoundary name="영상 분석">
                  <Suspense
                    fallback={
                      <div className="loading">영상 분석 도구를 여는 중…</div>
                    }
                  >
                    <ViewerPage api={api} />
                  </Suspense>
                </ScreenErrorBoundary>
              )}
              {page === "predictions" && (
                <ScreenErrorBoundary name="예측 비교">
                  <Suspense
                    fallback={
                      <div className="loading">예측 비교 도구를 여는 중…</div>
                    }
                  >
                    <PredictionsPage api={api} />
                  </Suspense>
                </ScreenErrorBoundary>
              )}
              {page === "official" && (
                <ScreenErrorBoundary name="공식 안내">
                  <Suspense
                    fallback={
                      <div className="loading">공식 안내를 여는 중…</div>
                    }
                  >
                    <OfficialPage api={api} competitions={competitions} />
                  </Suspense>
                </ScreenErrorBoundary>
              )}
              {page === "overview" && (
                <>
                  <div className="overview-strip">
                    <span>
                      <span className="status-dot" />내 대회 리서치 홈
                    </span>
                    <span>리더보드 관측 {date(brief.data_as_of)} KST</span>
                  </div>
                  <div className="metrics-grid">
                    <div className="metric-card featured">
                      <div className="metric-label">
                        공개 노트북 <BookOpen size={18} />
                      </div>
                      <div className="metric-number">
                        {brief.competition_source_counts?.notebook || 0}
                        <span>개</span>
                      </div>
                      <div className="metric-foot">
                        현재 대회 · 과거 대회 포함{" "}
                        {brief.source_counts?.notebook || 0}개
                      </div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">
                        대회 토론 <Lightbulb size={18} />
                      </div>
                      <div className="metric-number">
                        {brief.competition_source_counts?.discussion || 0}
                        <span>개</span>
                      </div>
                      <div className="metric-foot">
                        원글·댓글 수집 · 과거 대회 포함{" "}
                        {(brief.source_counts?.discussion || 0) +
                          (brief.source_counts?.solution || 0)}
                        개
                      </div>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">
                        내 최고 Public 점수 <Trophy size={18} />
                      </div>
                      <div className="metric-number">
                        {score(brief.best_score)}
                      </div>
                      {brief.leaderboard_summary?.own && (
                        <div className="metric-foot">
                          공개 LB{" "}
                          {brief.leaderboard_summary.own.rank.toLocaleString(
                            "ko-KR",
                          )}
                          위 /{" "}
                          {brief.leaderboard_summary.teams.toLocaleString(
                            "ko-KR",
                          )}
                          팀
                        </div>
                      )}
                      {brief.leaderboard_summary?.medals?.length > 0 && (
                        <div
                          className="metric-foot"
                          title={brief.leaderboard_summary.medal_note}
                        >
                          참고선{" "}
                          {brief.leaderboard_summary.medals
                            .map(
                              (m: RecordData) =>
                                ({ gold: "금", silver: "은", bronze: "동" })[
                                  m.medal as "gold"
                                ] +
                                " " +
                                score(m.score),
                            )
                            .join(" · ")}
                        </div>
                      )}
                      <button
                        className="text-button"
                        onClick={() => nav("submissions")}
                      >
                        제출 이력 보기 <ArrowRight size={13} />
                      </button>
                    </div>
                    <div className="metric-card">
                      <div className="metric-label">
                        최종 제출까지 <Clock3 size={18} />
                      </div>
                      <div className="metric-number">
                        {days == null ? "—" : `D−${days}`}
                      </div>
                      <div className="metric-foot">
                        {date(brief.competition.deadline)} KST
                      </div>
                    </div>
                  </div>
                  <div className="quick-links">
                    <a
                      href="https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard"
                      target="_blank"
                      rel="noreferrer"
                    >
                      <Trophy size={18} />
                      <span>
                        <strong>전체 리더보드</strong>
                        <small>Kaggle에서 최신 순위 확인</small>
                      </span>
                      <ArrowUpRight size={18} />
                    </a>
                    <a
                      href="https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions"
                      target="_blank"
                      rel="noreferrer"
                    >
                      <Upload size={18} />
                      <span>
                        <strong>Kaggle 제출로 이동</strong>
                        <small>노트북 제출 · 최종 제출 선택</small>
                      </span>
                      <ArrowUpRight size={18} />
                    </a>
                  </div>
                  <div className="overview-columns">
                    <section className="panel">
                      <div className="panel-title">
                        <div>
                          <h2>연구 자료 둘러보기</h2>
                          <p>
                            관심 있는 자료를 모으고, 원문과 버전을 확인하세요.
                          </p>
                        </div>
                        <button
                          className="text-button"
                          onClick={() => nav("research")}
                        >
                          라이브러리 <ArrowRight size={15} />
                        </button>
                      </div>
                      {[
                        {
                          kind: "notebook",
                          title: "공개 노트북",
                          text: "모델 구조 · 추론 코드 · 입력 데이터",
                          icon: BookOpen,
                        },
                        {
                          kind: "discussion",
                          title: "대회 토론",
                          text: "호스트 안내 · 실험 공유 · 최신 논점",
                          icon: Lightbulb,
                        },
                        {
                          kind: "official",
                          title: "공식 평가와 규칙",
                          text: "마감 · 평가 지표 · 제출 제약",
                          icon: ShieldCheck,
                        },
                      ].map((item) => (
                        <button
                          className="resource-row"
                          key={item.kind}
                          onClick={() => {
                            setKind(item.kind);
                            setQ("");
                            setScope("rsna-knee-abnormality-detection");
                            setOffset(0);
                            nav(
                              item.kind === "notebook"
                                ? "notebooks"
                                : item.kind === "discussion"
                                  ? "discussions"
                                  : "official",
                            );
                          }}
                        >
                          <span className={"resource-icon " + item.kind}>
                            <item.icon size={20} />
                          </span>
                          <span>
                            <strong>{item.title}</strong>
                            <small>{item.text}</small>
                          </span>
                          <ArrowUpRight size={18} />
                        </button>
                      ))}
                      <div className="interest-row">
                        <span>관심 소견</span>
                        <div className="tags">
                          {prefs.watch_targets.map((t: string) => (
                            <button
                              className="tag"
                              key={t}
                              onClick={() => {
                                setQ(t);
                                setKind("");
                                setScope("");
                                setOffset(0);
                                nav("research");
                              }}
                            >
                              {t}
                            </button>
                          ))}
                        </div>
                      </div>
                    </section>
                    {!prefs.hidden_cards.includes("leaderboard") && (
                      <section className="panel">
                        <div className="panel-title">
                          <h2>Public 리더보드</h2>
                          <Badge>상위 {brief.leaderboard.length}팀 수집</Badge>
                        </div>
                        <div className="leader-list">
                          {brief.leaderboard
                            .slice(0, 7)
                            .map((r: RecordData) => (
                              <div className="leader-row" key={r.rank}>
                                <span className="rank">
                                  {String(r.rank).padStart(2, "0")}
                                </span>
                                <span>{r.team_name}</span>
                                <strong>{score(r.score)}</strong>
                              </div>
                            ))}
                        </div>
                        <div className="panel-footer">
                          {date(brief.leaderboard[0]?.observed_at)} KST 관측 ·
                          전체 순위 아님
                        </div>
                      </section>
                    )}
                  </div>
                  <section className="panel">
                    <div className="panel-title">
                      <div>
                        <h2>최근 모은 자료</h2>
                        <p>
                          최근 관측 순 · 목록만 수집된 자료는 본문 수집을
                          눌러주세요.
                        </p>
                      </div>
                      <button
                        className="text-button"
                        onClick={() => {
                          setQ("");
                          setKind("");
                          setScope("");
                          setOffset(0);
                          nav("research");
                        }}
                      >
                        전체 자료 <ArrowRight size={15} />
                      </button>
                    </div>
                    <div className="recent-sources">
                      {brief.recent_sources?.map((s: RecordData) => (
                        <button key={s.id} onClick={() => openSource(s.id)}>
                          <Badge
                            tone={
                              s.authority === "official" ? "teal" : "neutral"
                            }
                          >
                            {kinds[s.kind]}
                          </Badge>
                          <strong>{s.title}</strong>
                          <span>{date(s.observed_at)}</span>
                          <ArrowUpRight size={15} />
                        </button>
                      ))}
                    </div>
                  </section>
                </>
              )}

              {page === "history" && (
                <>
                  <div className="alert">
                    <BookOpen size={19} />
                    <span>
                      노트북의 점수 정렬 목록과 입상 솔루션은 별개의 근거입니다.
                      토론 제목의 순위는 작성자 주장으로 표시합니다.
                    </span>
                  </div>
                  <div className="history-grid">
                    {competitions.map((c) => (
                      <section className="panel history-card" key={c.slug}>
                        <div className="card-top">
                          <Badge>
                            {c.count ? `${c.count}개 자료` : "수집 전"}
                          </Badge>
                          <a
                            href={
                              "https://www.kaggle.com/competitions/" + c.slug
                            }
                            target="_blank"
                            rel="noreferrer"
                            aria-label={c.title + " 원문"}
                          >
                            <ExternalLink size={17} />
                          </a>
                        </div>
                        <h2>{c.title}</h2>
                        <p>
                          {c.relevance ||
                            "공식 평가와 입력 데이터의 차이를 확인하며 참고하세요."}
                        </p>
                        <div className="history-stats">
                          <span>
                            솔루션 원문{" "}
                            <strong>
                              {c.snapshot?.solution_bodies == null
                                ? c.snapshot
                                  ? "재수집 필요"
                                  : "—"
                                : `${c.snapshot.solution_bodies}/${c.snapshot.solutions}`}
                            </strong>
                          </span>
                          <span>
                            토론{" "}
                            <strong>{c.snapshot?.discussions ?? "—"}</strong>
                          </span>
                          <span>
                            노트북 목록{" "}
                            <strong>{c.snapshot?.notebooks ?? "—"}</strong>
                          </span>
                        </div>
                        <small>
                          최근 수집 {date(c.snapshot?.observed_at)} KST ·{" "}
                          {c.snapshot?.notebook_sort === "scoreAscending"
                            ? "낮은 점수순"
                            : c.snapshot?.notebook_sort === "scoreDescending"
                              ? "높은 점수순"
                              : c.snapshot?.notebook_sort === "voteCount"
                                ? "추천순"
                                : "정렬 확인 전"}
                        </small>
                        <div className="dialog-actions">
                          <button
                            className="button primary"
                            disabled={working}
                            onClick={() => start("history", c.slug)}
                          >
                            <RefreshCw size={15} />
                            {c.count ? "다시 수집" : "자료 수집"}
                          </button>
                          <button
                            className="button secondary"
                            disabled={!c.count}
                            onClick={() => {
                              setScope(c.slug);
                              setQ("");
                              setKind("solution");
                              setOffset(0);
                              nav("solutions");
                            }}
                          >
                            솔루션 보기
                          </button>
                          <button
                            className="text-button"
                            disabled={!c.count}
                            onClick={() => {
                              setScope(c.slug);
                              setQ("");
                              setKind("");
                              setOffset(0);
                              nav("research");
                            }}
                          >
                            전체 자료 <ArrowRight size={14} />
                          </button>
                        </div>
                        {c.snapshot?.errors?.length > 0 && (
                          <p className="history-warning">
                            일부 수집 실패 · 내 작업 환경에서 원인 확인 후
                            재시도할 수 있습니다.
                          </p>
                        )}
                      </section>
                    ))}
                  </div>
                  <section className="panel settings-panel add-competition">
                    <h2>참고할 대회 추가</h2>
                    <p>
                      Kaggle 대회 주소의 마지막 부분을 입력하세요. 예:
                      rsna-2024-lumbar-spine-degenerative-classification
                    </p>
                    <div className="competition-form">
                      <label className="field">
                        대회 이름
                        <input
                          value={newTitle}
                          onChange={(e) => setNewTitle(e.target.value)}
                          placeholder="라이브러리에 표시할 이름"
                        />
                      </label>
                      <label className="field">
                        대회 URL 이름 (slug)
                        <input
                          value={newSlug}
                          onChange={(e) => setNewSlug(e.target.value)}
                          placeholder="competition-slug"
                        />
                      </label>
                      <button
                        className="button primary"
                        disabled={!newSlug || !newTitle}
                        onClick={() =>
                          guard(async () => {
                            setCompetitions(
                              await api("/competitions", {
                                method: "PUT",
                                body: JSON.stringify([
                                  ...competitions.map((c) => ({
                                    slug: c.slug,
                                    title: c.title,
                                    relevance: c.relevance,
                                  })),
                                  {
                                    slug: newSlug.trim(),
                                    title: newTitle.trim(),
                                    relevance: "",
                                  },
                                ]),
                              }),
                            );
                            setNewSlug("");
                            setNewTitle("");
                            notice(
                              "대회를 추가했습니다. 자료 수집 버튼으로 시작하세요.",
                            );
                          })
                        }
                      >
                        <Plus size={16} />
                        추가
                      </button>
                    </div>
                  </section>
                  <p className="footnote">
                    회당 전체 토론 목록을 확인하고 순위가 높은 솔루션부터 최대
                    40개, 인기 토론 10개의 원글과 댓글 전체를 받습니다. 이미 받은
                    본문은 댓글이 늘었을 때만 다시 받습니다. 노트북은 목록 50개와
                    새 소스 5개입니다. 원문 인용을 검증·전처리·모델·앙상블·효율
                    항목으로 찾아줍니다.
                  </p>
                </>
              )}

              {isLibrary && (
                <>
                  <SyncControl
                    api={api}
                    refreshKey={jobsKey}
                    onChange={scheduleChanged}
                  />
                  <div className="toolbar">
                    <div className="search-field">
                      <Search size={18} />
                      <input
                        aria-label="연구 자료 검색"
                        value={q}
                        onChange={(e) => {
                          setQ(e.target.value);
                          setOffset(0);
                        }}
                        placeholder="제목, 본문, 모델 이름으로 검색…"
                      />
                      {q && (
                        <button
                          className="icon-button"
                          onClick={() => setQ("")}
                          aria-label="검색 지우기"
                        >
                          <X size={15} />
                        </button>
                      )}
                    </div>
                    <select
                      aria-label="대회 선택"
                      value={scope}
                      onChange={(e) => {
                        setScope(e.target.value);
                        setOffset(0);
                      }}
                    >
                      <option value="">모든 대회</option>
                      <option value="rsna-knee-abnormality-detection">
                        RSNA Knee · 현재
                      </option>
                      {competitions.map((c) => (
                        <option key={c.slug} value={c.slug}>
                          {c.title}
                        </option>
                      ))}
                    </select>
                    <span className="muted">{sources.total}개 자료</span>
                  </div>
                  <div className="library-filters">
                    <label>
                      정렬
                      <select
                        aria-label="정렬"
                        value={sort}
                        onChange={(e) => {
                          setSort(e.target.value);
                          setOffset(0);
                        }}
                      >
                        <option value="added">처음 추가된 순</option>
                        <option value="activity">
                          {page === "notebooks"
                            ? "최근 실행 순"
                            : page === "discussions"
                              ? "최근 게시 순"
                              : "최근 활동 순"}
                        </option>
                        <option value="votes">추천 많은 순</option>
                        {page !== "discussions" && (
                          <option value="score">공개 점수 순 (확인됨)</option>
                        )}
                        {page !== "notebooks" && (
                          <option value="comments">댓글 많은 순</option>
                        )}
                        {page !== "discussions" && (
                          <option value="claim">제목 점수 표기 순</option>
                        )}
                        <option value="title">제목 순</option>
                      </select>
                    </label>
                    <label>
                      본문
                      <select
                        aria-label="본문 상태"
                        value={bodyFilter}
                        onChange={(e) => {
                          setBodyFilter(e.target.value);
                          setOffset(0);
                        }}
                      >
                        <option value="">전체</option>
                        <option value="full">코드·본문 있음</option>
                        <option value="listing">목록만</option>
                      </select>
                    </label>
                    {inputRef && (
                      <span className="filter-chip" title={inputRef}>
                        <span>입력: {inputRef}</span>
                        <button
                          aria-label="입력 필터 해제"
                          onClick={() => {
                            setInputRef("");
                            setOffset(0);
                          }}
                        >
                          <X size={14} />
                        </button>
                      </span>
                    )}
                  </div>
                  {page === "notebooks" && topInputs?.items?.length > 0 && (
                    <details className="top-inputs">
                      <summary>
                        자주 쓰인 입력 · 입력 기록이 있는 노트북{" "}
                        {topInputs?.notebooks_with_inputs}개 기준
                      </summary>
                      <p className="muted-note">
                        같은 가중치·라벨을 공유하는 계열을 찾는 데 쓰세요. 누르면
                        그 입력을 쓰는 노트북만 보여줍니다. 숫자는 사용 노트북 수와
                        그중 최고 공개 점수(확인됨) 또는 제목 표기(주장)입니다.
                      </p>
                      <ol>
                        {topInputs?.items.map((i: RecordData) => (
                          <li key={i.kind + i.ref}>
                            <button
                              onClick={() => {
                                setInputRef(i.ref);
                                setOffset(0);
                              }}
                              title={i.kind + " · " + i.ref}
                            >
                              <span>{i.ref}</span>
                              <small>
                                {i.n}
                                {i.best_score > 0
                                  ? " · 공개 " + i.best_score.toFixed(3)
                                  : i.best_claim > 0
                                    ? " · 제목 " + i.best_claim.toFixed(3)
                                    : ""}
                              </small>
                            </button>
                          </li>
                        ))}
                      </ol>
                    </details>
                  )}
                  <div
                    className="view-switch"
                    role="group"
                    aria-label="자료 표시 방식"
                  >
                    <button
                      aria-label="목록 보기"
                      aria-pressed={prefs.research_view !== "cards"}
                      className={
                        prefs.research_view !== "cards" ? "selected" : ""
                      }
                      onClick={() =>
                        guard(async () => {
                          const next = { ...prefs, research_view: "list" };
                          setPrefs(next);
                          await api("/preferences", {
                            method: "PUT",
                            body: JSON.stringify(next),
                          });
                        })
                      }
                    >
                      <List size={19} />
                      목록
                    </button>
                    <button
                      aria-label="카드 보기"
                      aria-pressed={prefs.research_view === "cards"}
                      className={
                        prefs.research_view === "cards" ? "selected" : ""
                      }
                      onClick={() =>
                        guard(async () => {
                          const next = { ...prefs, research_view: "cards" };
                          setPrefs(next);
                          await api("/preferences", {
                            method: "PUT",
                            body: JSON.stringify(next),
                          });
                        })
                      }
                    >
                      <LayoutGrid size={18} />
                      카드
                    </button>
                  </div>
                  {prefs.research_view === "cards" ? (
                    <div className="research-grid">
                      {sources.items.map((s: RecordData) => (
                        <button
                          className="research-card"
                          key={s.id}
                          onClick={() => openSource(s.id)}
                        >
                          <div className="card-top">
                            <Badge
                              tone={
                                s.authority === "official" ? "teal" : "neutral"
                              }
                            >
                              {kinds[s.kind] || s.kind}
                            </Badge>
                            {isToday(s.added_at) && (
                              <span className="new-badge">NEW</span>
                            )}
                            <ArrowUpRight size={17} />
                          </div>
                          <h3>{s.title}</h3>
                          <MetaChips meta={s} kind={s.kind} />
                          <p>{s.excerpt.slice(0, 125)}</p>
                          <div className="tags">
                            {s.tags.slice(0, 3).map((t: string) => (
                              <span className="tag" key={t}>
                                {t}
                              </span>
                            ))}
                          </div>
                          <div className="card-bottom">
                            <span>추가 {addedDate(s.added_at)}</span>
                            <span>{s.revision_count} 저장본</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="panel table-wrap source-table">
                      <table className="card-table">
                        <thead>
                          <tr>
                            <th>자료 / 출처</th>
                            <th>반응</th>
                            <th>
                              {page === "notebooks" ? "마지막 실행" : "게시·실행"}
                            </th>
                            <th>추가된 날짜 (KST)</th>
                            <th>상태</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sources.items.map((s: RecordData) => (
                            <tr key={s.id}>
                              <td className="source-title no-label">
                                <button onClick={() => openSource(s.id)}>
                                  <span className="source-title-text">
                                    {s.title}
                                  </span>
                                  {isToday(s.added_at) && (
                                    <span className="new-badge">NEW</span>
                                  )}
                                </button>
                                <div className="source-row-meta">
                                  <Badge
                                    tone={
                                      s.authority === "official"
                                        ? "teal"
                                        : "neutral"
                                    }
                                  >
                                    {kinds[s.kind] || s.kind}
                                  </Badge>
                                  <span>
                                    {s.competition === CURRENT
                                      ? "RSNA Knee · 현재"
                                      : competitions.find(
                                          (c) => c.slug === s.competition,
                                        )?.title || s.competition}
                                  </span>
                                  {s.public_score != null ? (
                                    <span
                                      className="score-badge"
                                      title="Kaggle 노트북 목록이 표시하는 best public score입니다. 비공식 웹 API로 관측했습니다."
                                    >
                                      공개 {s.public_score.toFixed(3)}
                                      {s.medal ? " · " + s.medal.toLowerCase() : ""}
                                    </span>
                                  ) : (
                                    s.score_claim > 0 && (
                                      <span
                                        className="claim-badge"
                                        title="제목에 적힌 점수입니다. 공개 점수가 확인되지 않았습니다."
                                      >
                                        제목 {s.score_claim.toFixed(3)}
                                      </span>
                                    )
                                  )}
                                  {s.tags.slice(0, 2).map((t: string) => (
                                    <span className="tag" key={t}>
                                      {t}
                                    </span>
                                  ))}
                                </div>
                              </td>
                              <td className="num-cell" data-label="반응">
                                {s.votes != null ? `추천 ${s.votes}` : "—"}
                                {s.comments != null && (
                                  <small>댓글 {s.comments}</small>
                                )}
                              </td>
                              <td className="num-cell" data-label="게시·실행">
                                {shortDate(s.last_run || s.posted_at)}
                              </td>
                              <td className="date-cell" data-label="추가">
                                <time dateTime={s.added_at}>
                                  {addedDate(s.added_at)}
                                </time>
                              </td>
                              <td className="version-cell" data-label="상태">
                                <span
                                  className={
                                    "state-pill " +
                                    (s.body_state === "full" ? "full" : "")
                                  }
                                >
                                  {s.body_state === "full"
                                    ? s.kind === "notebook" ||
                                      s.kind === "own_notebook"
                                      ? "코드 있음"
                                      : "본문 있음"
                                    : s.body_state === "listing"
                                      ? "목록만"
                                      : "일부"}
                                </span>
                                <small>
                                  {s.version && s.version !== "unknown"
                                    ? "v" + s.version
                                    : "snapshot"}
                                  {" · 저장본 " + s.revision_count}
                                </small>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {sources.total === 0 && (
                    <Empty
                      title="검색 결과가 없습니다"
                      body="다른 키워드를 사용하거나 자료를 먼저 불러오세요."
                    />
                  )}
                  <div className="pagination">
                    <button
                      className="button secondary"
                      disabled={offset === 0}
                      onClick={() => setOffset(Math.max(0, offset - 80))}
                    >
                      이전
                    </button>
                    <span>
                      {Math.min(offset + 1, sources.total)}–
                      {Math.min(offset + 80, sources.total)} / {sources.total}
                    </span>
                    <button
                      className="button secondary"
                      disabled={offset + 80 >= sources.total}
                      onClick={() => setOffset(offset + 80)}
                    >
                      다음
                    </button>
                  </div>
                </>
              )}

              {page === "experiments" && (
                <>
                  <div className="alert">
                    <ShieldCheck size={19} />
                    <span>
                      점수의 종류와 검증 범위를 먼저 확인하세요. 라벨·분할
                      정보가 없는 기록은 비교를 보류합니다.
                    </span>
                  </div>
                  <div className="experiment-filters">
                    <div className="search-field compact">
                      <Search size={16} />
                      <input
                        aria-label="실험 검색"
                        value={runQ}
                        onChange={(e) => setRunQ(e.target.value)}
                        placeholder="실험 이름·모델 계열"
                      />
                    </div>
                    <select
                      aria-label="지표 종류"
                      value={runKind}
                      onChange={(e) => setRunKind(e.target.value)}
                    >
                      <option value="">모든 지표 종류</option>
                      {Object.entries(metrics).map(([k, v]) => (
                        <option key={k} value={k}>
                          {v}
                        </option>
                      ))}
                    </select>
                    <select
                      aria-label="target 범위"
                      value={runScope}
                      onChange={(e) => setRunScope(e.target.value)}
                    >
                      <option value="">모든 target 범위</option>
                      <option value="12">12 target</option>
                      <option value="partial">일부 target</option>
                      <option value="scored">점수 있는 기록만</option>
                    </select>
                    <select
                      aria-label="실험 정렬"
                      value={runSort}
                      onChange={(e) => setRunSort(e.target.value)}
                    >
                      <option value="name">이름 순</option>
                      <option value="score">점수 높은 순</option>
                      <option value="observed">최근 관측 순</option>
                    </select>
                  </div>
                  <div className="toolbar">
                    <strong>
                      {shownRuns.length === runs.length
                        ? `${runs.length}개 실험`
                        : `${shownRuns.length} / ${runs.length}개 실험`}
                    </strong>
                    <span className="muted">
                      두 개를 선택하면 검증 조건을 비교합니다.
                    </span>
                    <button
                      className="button primary push-right"
                      disabled={chosen.length !== 2}
                      onClick={() =>
                        guard(async () =>
                          setComparison(
                            await api(
                              `/runs/compare?a=${chosen[0]}&b=${chosen[1]}`,
                            ),
                          ),
                        )
                      }
                    >
                      <GitCompareArrows size={16} />
                      선택 비교 ({chosen.length}/2)
                    </button>
                  </div>
                  {comparison && (
                    <div
                      className={
                        "panel compare-panel " +
                        (comparison.comparable ? "compatible" : "")
                      }
                    >
                      <div className="panel-title">
                        <h2>
                          {comparison.comparable
                            ? "비교 조건 일치"
                            : "직접 비교 보류"}
                        </h2>
                        <button
                          className="icon-button"
                          aria-label="비교 닫기"
                          onClick={() => setComparison(null)}
                        >
                          <X size={18} />
                        </button>
                      </div>
                      <p>
                        {comparison.records
                          .map((r: RecordData) => r.name)
                          .join(" ↔ ")}
                      </p>
                      {comparison.comparable ? (
                        <strong>
                          선택한 두 번째 실험 − 첫 번째 실험:{" "}
                          {comparison.delta?.toFixed(6)}
                        </strong>
                      ) : (
                        <ul>
                          {comparison.reasons.map((r: string) => (
                            <li key={r}>{r}</li>
                          ))}
                        </ul>
                      )}
                    </div>
                  )}
                  <div className="panel table-wrap">
                    <table className="card-table">
                      <thead>
                        <tr>
                          <th aria-label="선택"></th>
                          <th>실험 / 모델</th>
                          <th>지표 종류</th>
                          <th>점수</th>
                          <th>범위</th>
                          <th>관측</th>
                        </tr>
                      </thead>
                      <tbody>
                        {shownRuns.map((r) => (
                          <tr key={r.id}>
                            <td className="no-label">
                              <input
                                type="checkbox"
                                aria-label={r.name + " 선택"}
                                checked={chosen.includes(r.id)}
                                disabled={
                                  !chosen.includes(r.id) && chosen.length === 2
                                }
                                onChange={(e) =>
                                  setChosen(
                                    e.target.checked
                                      ? [...chosen, r.id]
                                      : chosen.filter((id) => id !== r.id),
                                  )
                                }
                              />
                            </td>
                            <td className="run-name no-label">
                              <strong>{r.name}</strong>
                              <small>{r.family}</small>
                              <details>
                                <summary>근거와 검증 조건</summary>
                                <p>
                                  {r.warnings.join(" ") ||
                                    "기록된 검증 조건을 확인하세요."}
                                </p>
                                <pre>{JSON.stringify(r.contract, null, 2)}</pre>
                                <small>{r.uri || "경로 정보 없음"}</small>
                              </details>
                            </td>
                            <td data-label="지표 종류">
                              <Badge
                                tone={
                                  r.kind === "pseudo_oof_leaky"
                                    ? "amber"
                                    : "neutral"
                                }
                              >
                                {metrics[r.kind] || r.kind}
                              </Badge>
                            </td>
                            <td className="number" data-label="점수">
                              {r.score == null ? "—" : r.score.toFixed(5)}
                            </td>
                            <td data-label="범위">
                              <strong>{r.targets.length} targets</strong>
                              <small>
                                {r.rows_count?.toLocaleString() || "?"} studies
                              </small>
                            </td>
                            <td className="muted nowrap" data-label="관측">
                              {date(r.observed_at)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {runs.length === 0 && (
                      <Empty
                        title="실험 기록이 없습니다"
                        body="로컬 자료 읽기로 실험 요약 파일을 가져오세요."
                      />
                    )}
                  </div>
                </>
              )}

              {page === "submissions" && (
                <>
                  <div className="submission-header panel">
                    <div>
                      <span className="eyebrow">VERIFIED PUBLIC SCORE</span>
                      <h2>
                        {score(brief.best_score)} <span>내 최고 점수</span>
                      </h2>
                      {brief.leaderboard_summary?.own && (
                        <p>
                          공개 LB{" "}
                          {brief.leaderboard_summary.own.rank.toLocaleString("ko-KR")}
                          위 /{" "}
                          {brief.leaderboard_summary.teams.toLocaleString("ko-KR")}
                          팀 · 상위 {brief.leaderboard_summary.own.percentile}%
                        </p>
                      )}
                    </div>
                    <div>
                      <strong>
                        오늘 남은 제출{" "}
                        {brief.submission_limits?.remaining_today ?? "—"} /{" "}
                        {brief.competition.max_daily_submissions ?? "—"}회
                      </strong>
                      <p>
                        {brief.submission_limits
                          ? `관측 ${date(brief.submission_limits.observed_at)} KST · UTC 0시(KST 9시) 초기화`
                          : "Kaggle 새로고침 후 표시됩니다"}
                      </p>
                    </div>
                    <div>
                      <strong>최종 제출 마감</strong>
                      <p>{date(brief.competition.deadline)} KST</p>
                    </div>
                  </div>
                  <div className="panel table-wrap">
                    <table className="card-table">
                      <thead>
                        <tr>
                          <th>제출 내용</th>
                          <th>Public AUC</th>
                          <th>상태</th>
                          <th>제출 시간 (KST)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {brief.submissions.map((r: RecordData) => (
                          <tr key={r.id}>
                            <td className="submission-name no-label">
                              <strong>{r.description || "설명 없음"}</strong>
                              <small>#{r.id}</small>
                            </td>
                            <td className="number" data-label="Public AUC">
                              {score(r.score)}
                            </td>
                            <td data-label="상태">
                              <Badge
                                tone={
                                  String(r.status)
                                    .toUpperCase()
                                    .includes("COMPLETE")
                                    ? "teal"
                                    : "amber"
                                }
                              >
                                {String(r.status).replace(
                                  "SubmissionStatus.",
                                  "",
                                )}
                              </Badge>
                            </td>
                            <td className="nowrap muted" data-label="제출 시간">
                              {date(r.submitted_at)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="footnote">
                    제출 기록을 조회합니다. 새 제출과 최종 제출 선택은
                    Kaggle에서 진행하세요.
                  </p>
                </>
              )}

              {page === "settings" && (
                <div className="settings-grid">
                  <section className="panel settings-panel">
                    <div className="panel-title">
                      <h2>
                        <SlidersHorizontal size={19} />
                        화면과 관심 소견
                      </h2>
                    </div>
                    <label className="field">
                      화면 테마
                      <select
                        value={prefs.theme}
                        onChange={(e) =>
                          setPrefs({ ...prefs, theme: e.target.value })
                        }
                      >
                        <option value="light">밝게</option>
                        <option value="dark">어둡게</option>
                      </select>
                    </label>
                    <label className="check-field">
                      <input
                        type="checkbox"
                        checked={prefs.compact}
                        onChange={(e) =>
                          setPrefs({ ...prefs, compact: e.target.checked })
                        }
                      />
                      촘촘한 목록 간격
                    </label>
                    <div className="field">
                      관심 소견
                      <div className="target-options">
                        {session.targets.map((t: string) => (
                          <label key={t}>
                            <input
                              type="checkbox"
                              checked={prefs.watch_targets.includes(t)}
                              onChange={(e) =>
                                setPrefs({
                                  ...prefs,
                                  watch_targets: e.target.checked
                                    ? [...prefs.watch_targets, t]
                                    : prefs.watch_targets.filter(
                                        (v: string) => v !== t,
                                      ),
                                })
                              }
                            />
                            {t}
                          </label>
                        ))}
                      </div>
                    </div>
                    {[["leaderboard", "홈에서 리더보드 숨기기"]].map(
                      ([key, label]) => (
                        <label className="check-field" key={key}>
                          <input
                            type="checkbox"
                            checked={prefs.hidden_cards.includes(key)}
                            onChange={(e) =>
                              setPrefs({
                                ...prefs,
                                hidden_cards: e.target.checked
                                  ? [...prefs.hidden_cards, key]
                                  : prefs.hidden_cards.filter(
                                      (v: string) => v !== key,
                                    ),
                              })
                            }
                          />
                          {label}
                        </label>
                      ),
                    )}
                    <button
                      className="button primary"
                      onClick={savePreferences}
                    >
                      <Check size={16} />
                      개인 설정 저장
                    </button>
                  </section>
                  <section className="panel settings-panel">
                    <div className="panel-title">
                      <h2>
                        <RefreshCw size={19} />
                        수집과 상태
                      </h2>
                    </div>
                    <SyncControl
                      api={api}
                      refreshKey={jobsKey}
                      onChange={scheduleChanged}
                      showCoverage={false}
                    />
                    <div className="field">
                      함께 수집할 과거 대회{" "}
                      {competitions.map((c) => (
                        <label className="check-field" key={c.slug}>
                          <input
                            type="checkbox"
                            checked={
                              prefs.sync_history?.includes(c.slug) || false
                            }
                            onChange={(e) =>
                              setPrefs({
                                ...prefs,
                                sync_history: e.target.checked
                                  ? [...(prefs.sync_history || []), c.slug]
                                  : (prefs.sync_history || []).filter(
                                      (x: string) => x !== c.slug,
                                    ),
                              })
                            }
                          />
                          {c.title}
                        </label>
                      ))}
                    </div>
                    <p className="footnote">
                      정기 수집 켜기·간격은 바로 적용됩니다. 매번 리더보드·제출·
                      공식 안내와 전체 노트북·토론 목록을 확인하고, 새 자료와 댓글이
                      늘어난 토론, 실행 기록이 바뀐 노트북을 정해진 수만큼 받습니다.
                      인증은 각 PC에서 설정합니다.
                    </p>
                    <button
                      className="button secondary"
                      onClick={savePreferences}
                    >
                      과거 대회 선택 저장
                    </button>
                    <div className="connector-list">
                      {brief.connectors.map((c: RecordData) => (
                        <details key={c.name}>
                          <summary>
                            <span
                              className={
                                "status-dot " +
                                (c.status === "ok" ? "" : "warn")
                              }
                            />
                            <strong>
                              {connectorNames[c.name] ||
                                (c.name.startsWith("history:")
                                  ? competitions.find(
                                      (item) => item.slug === c.name.slice(8),
                                    )?.title || "과거 대회"
                                  : c.name)}
                            </strong>
                            <Badge tone={c.status === "ok" ? "teal" : "amber"}>
                              {c.status}
                            </Badge>
                          </summary>
                          <p>{c.detail}</p>
                          <p>최근 성공: {date(c.last_success)} KST</p>
                          <small>{c.coverage}</small>
                        </details>
                      ))}
                    </div>
                  </section>
                  <section className="panel settings-panel">
                    <div className="panel-title">
                      <h2>
                        <Database size={19} />
                        작업 기록과 백업
                      </h2>
                    </div>
                    <button
                      className="button secondary"
                      onClick={() =>
                        guard(async () => {
                          const r = await fetch("/api/backup", {
                            method: "POST",
                            headers: { "X-Kneeops-Token": token },
                          });
                          if (!r.ok) throw Error("백업에 실패했습니다.");
                          const url = URL.createObjectURL(await r.blob());
                          const a = document.createElement("a");
                          a.href = url;
                          a.download = "kneeops-backup.sqlite";
                          a.click();
                          setTimeout(() => URL.revokeObjectURL(url), 1000);
                          notice("현재 기록을 백업했습니다.");
                        })
                      }
                    >
                      <ArrowDownToLine size={16} />내 기록 전체 백업
                    </button>
                    <p className="footnote">
                      백업에는 수집한 자료, 개인 설정과 제출 이력이 포함됩니다.
                      복원 방법은 설치 안내에 있습니다.
                    </p>
                    <div className="job-list">
                      {jobs.map((j) => (
                        <details key={j.id}>
                          <summary>
                            <strong>{j.kind}</strong>
                            <Badge
                              tone={
                                j.status === "complete"
                                  ? "teal"
                                  : j.status === "failed"
                                    ? "amber"
                                    : "neutral"
                              }
                            >
                              {j.status}
                            </Badge>
                            <span>{date(j.started_at)}</span>
                          </summary>
                          <pre>{j.error || j.result || j.progress}</pre>
                        </details>
                      ))}
                    </div>
                  </section>
                </div>
              )}
            </>
          )}
          <footer className="page-footer">
            <span>KneeOps · 개인 대회 연구실</span>
            <span>출처를 남기고, 검증을 쌓고, 다음 실험으로.</span>
          </footer>
        </div>
      </main>
      {toast && (
        <div className="toast" role="status">
          <Check size={18} />
          {toast}
        </div>
      )}
      <dialog
        ref={sourceDialog}
        className="source-dialog"
        onCancel={() => setSelected(null)}
      >
        {selected && (
          <>
            <div className="dialog-header">
              <Badge>{kinds[selected.kind]}</Badge>
              <button
                className="icon-button"
                onClick={() => setSelected(null)}
                aria-label="자료 닫기"
              >
                <X />
              </button>
            </div>
            <h2>{selected.title}</h2>
            <div className="source-meta">
              <span>
                {selected.authority === "official"
                  ? "공식 안내"
                  : "참고 자료 · 내용은 개별 검증 필요"}
              </span>
              <span>추가 {addedDate(selected.added_at)} KST</span>
              <span>최근 관측 {date(selected.observed_at)} KST</span>
            </div>
            <MetaChips meta={selected.meta} kind={selected.kind} />
            <div className="dialog-actions">
              {selected.url && (
                <a
                  className="button secondary"
                  href={safeURL(selected.url)}
                  target="_blank"
                  rel="noreferrer"
                >
                  <ExternalLink size={15} />
                  원문
                </a>
              )}
              {["notebook", "discussion", "solution"].includes(
                selected.kind,
              ) && (
                <button
                  className="button secondary"
                  disabled={working}
                  onClick={() =>
                    start(
                      selected.kind === "notebook" ? "notebook" : "topic",
                      selected.id.slice(selected.id.indexOf(":") + 1),
                    )
                  }
                >
                  <RefreshCw size={15} />
                  본문 수집
                </button>
              )}
            </div>
            {selected.kind === "notebook" && (
              <div className="version-fetch">
                <label className="field">
                  특정 Kaggle 버전 수집
                  <input
                    type="number"
                    min="1"
                    step="1"
                    value={versionNumber}
                    onChange={(e) => setVersionNumber(e.target.value)}
                    placeholder="예: 3"
                  />
                </label>
                <button
                  className="button secondary"
                  disabled={
                    working || !versionNumber || Number(versionNumber) < 1
                  }
                  onClick={() =>
                    start(
                      "notebook",
                      selected.id.slice(9) + "/" + versionNumber,
                    )
                  }
                >
                  이 버전 가져오기
                </button>
                <button
                  className="text-button"
                  onClick={() => openSource(selected.id)}
                >
                  버전 목록 새로고침
                </button>
              </div>
            )}
            <label className="field">
              저장된 버전
              <select
                value={selected.revision.id}
                onChange={(e) => openSource(selected.id, e.target.value)}
              >
                {selected.revisions.map((r: RecordData) => (
                  <option key={r.id} value={r.id}>
                    {date(r.observed_at)} · {r.version || "snapshot"} ·{" "}
                    {r.sha.slice(0, 10)}
                  </option>
                ))}
              </select>
            </label>
            {selected.revisions.length > 1 && (
              <button
                className="text-button"
                onClick={() =>
                  guard(async () =>
                    setDiff(
                      (
                        await api(
                          `/source/diff?id=${encodeURIComponent(selected.id)}&before=${selected.revisions[1].id}&after=${selected.revisions[0].id}`,
                        )
                      ).diff || "본문 변경 없음",
                    ),
                  )
                }
              >
                <GitCompareArrows size={16} />
                최근 두 버전 차이 보기
              </button>
            )}
            <div className="source-provenance">
              SHA256 {selected.revision.sha}
              <br />
              {selected.revision.payload.coverage ||
                selected.local_uri ||
                "수집된 원문 범위를 기준으로 표시합니다."}
            </div>
            <SourceInsights
              api={api}
              source={selected}
              onOpen={(id) => openSource(id)}
              onDiff={(text) => setDiff(text)}
              onFilterInput={(ref) => {
                setSelected(null);
                setQ("");
                setOffset(0);
                setInputRef(ref);
                nav("notebooks");
              }}
            />
            {selected.strategy_evidence?.length > 0 && (
              <details className="strategy-details">
                <summary>
                  전략 관련 원문 인용 {selected.strategy_evidence.length}개
                </summary>
                <p>
                  키워드로 찾은 원문입니다. 검증된 성능 요약이나 AI의 판단이
                  아닙니다.
                </p>
                {selected.strategy_evidence.map((e: RecordData, i: number) => (
                  <blockquote key={i}>
                    <Badge>{e.aspect}</Badge>
                    <span>원문 {e.line}행</span>
                    <p>{e.quote}</p>
                  </blockquote>
                ))}
              </details>
            )}
            {selected.kind === "official" && !diff ? (
              <ScreenErrorBoundary
                name="공식 문서"
                key={selected.id + selected.revision.id}
              >
                <Suspense
                  fallback={<div className="loading">문서를 여는 중…</div>}
                >
                  <OfficialDocument
                    key={selected.id + selected.revision.id}
                    source={selected}
                  />
                </Suspense>
              </ScreenErrorBoundary>
            ) : diff ? (
              <>
                <button className="text-button" onClick={() => setDiff("")}>
                  <X size={14} /> 차이 닫고 본문 보기
                </button>
                <DiffView text={diff} />
              </>
            ) : (
              <pre className="source-content">{selected.revision.content}</pre>
            )}
          </>
        )}
      </dialog>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <ScreenErrorBoundary>
    <App />
  </ScreenErrorBoundary>,
);
