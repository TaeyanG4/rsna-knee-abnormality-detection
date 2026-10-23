import React, { useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  Check,
  Database,
  ExternalLink,
  FolderOpen,
  HardDrive,
  LoaderCircle,
  RefreshCw,
  Square,
  X,
} from "lucide-react";
import "./data.css";
import RawDataPanel from "./RawDataPanel";
import CacheOrigin from "./CacheOrigin";

type Row = Record<string, any>;
type Api = (path: string, options?: RequestInit) => Promise<any>;
const size = (n?: number | null): string => {
  if (n == null) return "확인 필요";
  if (n >= 1024 ** 4) return `${(n / 1024 ** 4).toFixed(2)} TiB`;
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  return `${n.toLocaleString()} B`;
};
const stamp = (s: string) =>
  new Date(s).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" });
const modeNames: Row = {
  metadata: "CSV 메타데이터",
  sample: "공개 테스트 원본",
  selected: "선택 study 원본",
  all: "전체 원본 ZIP",
};
const statusNames: Row = {
  planned: "미리보기",
  queued: "대기 중",
  running: "다운로드 중",
  stopping: "중지 중",
  cancelled: "중지됨",
  interrupted: "앱 종료로 중단",
  failed: "실패",
  complete: "완료",
};
const post = (body?: unknown): RequestInit => ({
  method: "POST",
  body: body === undefined ? undefined : JSON.stringify(body),
});

export default function DataPage({ api }: { api: Api }) {
  const [data, setData] = useState<Row | null>(null);
  const [path, setPath] = useState("");
  const [cachePath, setCachePath] = useState("");
  const [jobs, setJobs] = useState<Row[]>([]);
  const [plan, setPlan] = useState<Row | null>(null);
  const [query, setQuery] = useState("");
  const [studies, setStudies] = useState<Row>({ items: [], total: 0 });
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [archiveAvailable, setArchiveAvailable] = useState(false);
  const sequence = useRef(0);
  const refresh = async () => {
    const d = await api("/data");
    setData(d);
    return d;
  };
  useEffect(() => {
    let alive = true;
    api("/data/library")
      .then(
        (d) =>
          alive &&
          setArchiveAvailable(
            d.archives?.some((a: Row) => a.available) || false,
          ),
      )
      .catch(() => {});
    api("/data")
      .then((d) => {
        if (!alive) return;
        setData(d);
        setPath(d.settings.destination);
        setCachePath(d.settings.cache_manifest);
      })
      .catch((e) => alive && setError(e.message));
    const poll = () =>
      api("/data/downloads")
        .then((j) => alive && setJobs(j))
        .catch((e) => alive && setError(e.message));
    poll();
    const timer = setInterval(poll, 2000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    const n = ++sequence.current;
    const timer = setTimeout(
      () =>
        api("/data/studies?q=" + encodeURIComponent(query))
          .then((r) => {
            if (n === sequence.current) setStudies(r);
          })
          .catch((e) => n === sequence.current && setError(e.message)),
      250,
    );
    return () => clearTimeout(timer);
  }, [query, data?.settings.destination]);
  const run = async (label: string, action: () => Promise<void>) => {
    setBusy(label);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  };
  const preview = (mode: string) =>
    run("파일 목록과 공간을 확인하고 있습니다", async () => {
      setPlan(null);
      const p = await api("/data/preview", post({ mode, studies: selected }));
      setPlan(p);
      setJobs(await api("/data/downloads"));
    });
  const connectCache = (manifest: string) =>
    run("캐시 구조를 확인하고 있습니다", async () => {
      await api("/data/cache", {
        method: "PUT",
        body: JSON.stringify({ manifest }),
      });
      setCachePath(manifest);
      await refresh();
      setNotice(
        "캐시를 연결했습니다. 학습 입력 JSON을 내보내 Codex·Claude 또는 학습 코드에서 참조하세요.",
      );
    });
  const active = jobs.some((j) =>
    ["queued", "running", "stopping"].includes(j.status),
  );
  const unsaved = data && path.trim() !== data.settings.destination;
  const disabled = !!busy || !!unsaved || active;
  if (!data)
    return (
      <div className="data-page">
        <div className="loading">
          {error || "데이터 환경을 확인하고 있습니다…"}
        </div>
      </div>
    );
  const valid = data.caches.filter((c: Row) => c.valid);
  const totalCache = valid.reduce((sum: number, c: Row) => sum + c.bytes, 0);

  return (
    <div className="data-page">
      <div className="data-panel data-inline">
        <div>
          <strong>내려받은 MRI와 캐시를 직접 확인하세요.</strong>
          <p className="data-hint">
            슬라이스 탐색·대비 조절·ROI 통계·공식 라벨·검토 메모
          </p>
        </div>
        <a className="button primary" href="#viewer">
          <FolderOpen size={18} />
          영상 분석 열기
        </a>
      </div>
      {error && (
        <div className="alert error" role="alert">
          {error}
          <button
            className="icon-button"
            aria-label="데이터 오류 닫기"
            onClick={() => setError("")}
          >
            <X size={18} />
          </button>
        </div>
      )}
      {notice && (
        <div className="alert" role="status">
          <Check size={18} />
          {notice}
        </div>
      )}
      {busy && (
        <div className="alert" role="status">
          <LoaderCircle size={18} className="spin" />
          {busy}…
        </div>
      )}

      <RawDataPanel api={api} />
      <section className="data-recommend">
        <div className="data-recommend-icon">
          <Database size={27} />
        </div>
        <div>
          <span className="eyebrow">권장 시작점 · CACHE FIRST</span>
          <h2>570 GB를 받기 전에, 기존 캐시부터 활용하세요.</h2>
          <p>
            {valid.length
              ? `${valid.length}개 캐시의 구조를 확인했습니다. 필요한 study만 메모리에 읽어 반복 학습하고, 추가 영상이 필요한 경우 원본을 선택해서 받으세요.`
              : "원본 전체가 없어도 전처리 캐시로 학습할 수 있습니다. 캐시가 있는 폴더를 연결하거나 Kaggle에서 먼저 생성하세요."}
          </p>
        </div>
        <a
          className="button secondary"
          href="#cache-section"
          onClick={(e) => {
            e.preventDefault();
            document
              .getElementById("cache-section")
              ?.scrollIntoView({ behavior: "smooth" });
          }}
        >
          캐시 확인 <ArrowDownToLine size={16} />
        </a>
      </section>

      <div className="data-metrics">
        <div>
          <span>대회 전체 원본</span>
          <strong>{size(data.official?.bytes)}</strong>
          <small>약 569.8 GB · 819,640개 파일 · 10/1 관측</small>
        </div>
        <div>
          <span>구조 확인된 로컬 캐시</span>
          <strong>{size(totalCache)}</strong>
          <small>{valid.length}개 폴더 합계 · 중복 후보 포함</small>
        </div>
        <div>
          <span>다운로드 드라이브 여유</span>
          <strong>{size(data.settings.free_bytes)}</strong>
          <small>실행 전 다시 확인합니다</small>
        </div>
      </div>

      <section className="data-panel">
        <div className="data-section-heading">
          <div>
            <h2>
              <FolderOpen size={20} /> 다운로드 저장 위치
            </h2>
            <p>이 PC에만 저장되는 설정입니다. 새 다운로드 계획에 적용됩니다.</p>
          </div>
          <span className="badge">개인 설정</span>
        </div>
        <form
          className="data-path-form"
          onSubmit={(e) => {
            e.preventDefault();
            run("경로를 저장하고 있습니다", async () => {
              const s = await api("/data/settings", {
                method: "PUT",
                body: JSON.stringify({ destination: path }),
              });
              setPath(s.destination);
              setData({ ...data, settings: s });
              setPlan(null);
              setNotice("다운로드 위치를 저장했습니다.");
            });
          }}
        >
          <label htmlFor="data-path">다운로드 폴더</label>
          <div className="data-path-row">
            <input
              id="data-path"
              value={path}
              placeholder="예: D:\KaggleData\rsna-knee"
              onChange={(e) => {
                setPath(e.target.value);
                setPlan(null);
              }}
            />
            <button className="button primary" disabled={!!busy}>
              경로 저장
            </button>
            <button
              className="button secondary"
              type="button"
              disabled={!!busy}
              onClick={() => {
                setPath(data.settings.default_destination);
                setPlan(null);
              }}
            >
              기본값
            </button>
          </div>
          <p className="data-hint">
            {unsaved
              ? "변경한 경로를 먼저 저장하세요."
              : "기존 파일은 덮어쓰지 않습니다. 다른 드라이브의 절대 경로 또는 작업 공간 기준 상대 경로를 입력할 수 있습니다."}
          </p>
        </form>
      </section>

      <section className="data-panel" id="cache-section">
        <div className="data-section-heading">
          <div>
            <h2>
              <Database size={20} /> 기존 캐시로 학습 준비
            </h2>
            <p>
              파일을 이동하거나 복사하지 않습니다. 원본 DICOM 전체가 없어도
              이용할 수 있습니다.
            </p>
          </div>
          <button
            className="button secondary"
            disabled={!!busy}
            onClick={() =>
              run("캐시를 다시 확인하고 있습니다", async () => {
                await refresh();
              })
            }
          >
            <RefreshCw size={16} /> 다시 검사
          </button>
        </div>
        <div className="data-cache-list">
          {data.caches.map((c: Row) => (
            <article className="data-cache-item" key={c.manifest}>
              <div className="data-cache-content">
                <div className="data-inline">
                  <strong>
                    {c.name.startsWith("EXP034")
                      ? "Adjacent · 4방향 × 9슬라이스 · 288px"
                      : c.name.startsWith("EXP032")
                        ? "Orthodiffusion · 3방향 × 16슬라이스 · 256px"
                        : c.name}
                  </strong>
                  <span className={c.valid ? "badge good" : "badge warn"}>
                    {c.valid ? "구조 확인" : "확인 필요"}
                  </span>
                </div>
                <p>
                  {c.studies?.toLocaleString()} study · {size(c.bytes)}{" "}
                  {c.same_manifest_content_elsewhere &&
                    "· 동일 해시 명세의 캐시가 다른 폴더에도 있음"}
                  {!!c.aliases?.length &&
                    "· 중복 실물 정리됨, 기존 경로 호환 유지"}
                </p>
                <code className="data-path-text">{c.manifest}</code>
                <CacheOrigin manifestSha={c.manifest_sha256} />
                {!c.valid && (
                  <p className="data-error-text">{c.errors.join(" · ")}</p>
                )}
              </div>
              <button
                className={
                  "button " +
                  (data.settings.cache_manifest === c.manifest
                    ? "secondary"
                    : "primary")
                }
                disabled={
                  !!busy ||
                  !c.valid ||
                  data.settings.cache_manifest === c.manifest
                }
                onClick={() => connectCache(c.manifest)}
              >
                {data.settings.cache_manifest === c.manifest ? (
                  <>
                    <Check size={16} /> 연결됨
                  </>
                ) : (
                  "이 캐시 연결"
                )}
              </button>
            </article>
          ))}
        </div>
        {!data.caches.length && (
          <p className="data-hint">
            자동으로 찾은 캐시가 없습니다. 아래에 지원되는 캐시 manifest 경로를
            입력하세요.
          </p>
        )}
        <details className="data-details">
          <summary>다른 폴더의 캐시 연결</summary>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              connectCache(cachePath);
            }}
          >
            <label htmlFor="cache-path">캐시 manifest JSON 경로</label>
            <div className="data-path-row">
              <input
                id="cache-path"
                value={cachePath}
                onChange={(e) => setCachePath(e.target.value)}
              />
              <button
                className="button secondary"
                disabled={!!busy || !cachePath.trim()}
              >
                검사 후 연결
              </button>
            </div>
            <p className="data-hint">
              현재 Adjacent / Orthodiffusion의 uint8 NPY shard·UID·mask 명세를
              지원합니다.
            </p>
          </form>
        </details>
        <div className="data-cache-footer">
          <p>
            NPY 헤더·파일 크기·UID·마스크를 검사합니다. 영상 전체 SHA256과 영상
            품질은 별도 검증이 필요합니다.
          </p>
          {data.settings.cache_manifest && (
            <a
              className="button secondary"
              href="/api/data/cache/descriptor"
              download
            >
              <ArrowDownToLine size={16} /> 학습 입력 JSON
            </a>
          )}
        </div>
      </section>

      <section className="data-panel">
        <div className="data-section-heading">
          <div>
            <h2>
              <ArrowDownToLine size={20} /> 필요한 만큼 다운로드
            </h2>
            <p>
              먼저 목록과 용량을 확인한 후 시작합니다. 로그인한 Kaggle 계정을
              사용합니다.
            </p>
          </div>
        </div>
        <div className="data-download-options">
          <article>
            <span className="data-option-step">01 · 가장 가벼운 시작</span>
            <h3>CSV 메타데이터</h3>
            <p>라벨·study·시리즈 목록과 제출 형식. 영상은 포함하지 않습니다.</p>
            <strong>5개 파일 · 약 8.7 MiB</strong>
            <button
              className="button secondary"
              disabled={disabled}
              onClick={() => preview("metadata")}
            >
              CSV 다운로드 준비
            </button>
          </article>
          <article>
            <span className="data-option-step">02 · 입출력 점검</span>
            <h3>공개 테스트 원본</h3>
            <p>
              공개된 테스트 study 전체를 받아 DICOM 읽기와 추론 경로를
              확인합니다.
            </p>
            <strong>공개 샘플 · 숨겨진 평가 데이터 제외</strong>
            <button
              className="button secondary"
              disabled={disabled}
              onClick={() => preview("sample")}
            >
              테스트 원본 다운로드 준비
            </button>
          </article>
          <article>
            <span className="data-option-step">03 · 전체가 필요할 때</span>
            <h3>전체 원본 ZIP</h3>
            <p>
              모든 데이터를 하나의 압축 파일로 받습니다. 완료 후 로컬 목록을
              새로고침하면 study별로 준비하여 열 수 있습니다.
            </p>
            <strong>
              원본 약 {size(data.official?.bytes)} · ZIP 용량은 다름
            </strong>
            <button
              className="button secondary"
              disabled={disabled}
              onClick={() => preview("all")}
            >
              전체 다운로드 준비
            </button>
          </article>
        </div>

        <details className="data-details data-study-picker">
          <summary>
            학습 study 선택 다운로드 <span>최대 5개씩 · 모든 시리즈 보존</span>
          </summary>
          <p>
            오류 분석·새 전처리에 필요한 study를 골라 받으세요. 파일명 순서로
            슬라이스를 잘라 받지 않습니다.
          </p>
          <label htmlFor="study-search">StudyInstanceUID 검색</label>
          <input
            id="study-search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="UID 일부로 검색"
          />
          <div className="data-selected">
            {selected.map((uid) => (
              <button
                key={uid}
                onClick={() => setSelected(selected.filter((x) => x !== uid))}
                title={uid}
              >
                {uid.slice(-16)} <X size={14} />
                <span className="sr-only">선택 해제</span>
              </button>
            ))}
          </div>
          <div className="data-study-list" aria-label="학습 study 목록">
            {studies.items.map((s: Row) => (
              <label key={s.uid}>
                <input
                  type="checkbox"
                  checked={selected.includes(s.uid)}
                  disabled={!selected.includes(s.uid) && selected.length >= 5}
                  onChange={(e) => {
                    setPlan(null);
                    setSelected(
                      e.target.checked
                        ? [...selected, s.uid]
                        : selected.filter((x) => x !== s.uid),
                    );
                  }}
                />
                <code>{s.uid}</code>
                <span>{s.series} 시리즈</span>
              </label>
            ))}
          </div>
          <div className="data-inline data-picker-footer">
            <p>
              {studies.metadata_available
                ? `검색 ${studies.total.toLocaleString()}개 중 최대 50개 표시 · ${selected.length}/5 선택`
                : "먼저 CSV 메타데이터를 내려받으세요."}
            </p>
            <button
              className="button primary"
              disabled={disabled || !selected.length}
              onClick={() => preview("selected")}
            >
              선택 원본 다운로드 준비
            </button>
          </div>
        </details>

        {plan && (
          <div
            className="data-preview"
            role="region"
            aria-label="다운로드 미리보기"
          >
            <div className="data-section-heading">
              <div>
                <h3>{modeNames[plan.mode]} · 다운로드 미리보기</h3>
                <p>
                  {plan.file_count.toLocaleString()}개 파일
                  {plan.studies.length ? ` · ${plan.studies.length} study` : ""}
                </p>
              </div>
              <button
                className="icon-button"
                aria-label="미리보기 닫기"
                onClick={() => setPlan(null)}
              >
                <X size={18} />
              </button>
            </div>
            <dl>
              <div>
                <dt>저장 위치</dt>
                <dd className="data-path-text">{plan.destination}</dd>
              </div>
              <div>
                <dt>
                  {plan.mode === "all" ? "원본 크기 기준" : "선택 원본 합계"}
                </dt>
                <dd>{size(plan.estimated_bytes)}</dd>
              </div>
              <div>
                <dt>권장 여유 공간 / 현재</dt>
                <dd>
                  {size(plan.required_free_bytes)} / {size(plan.free_bytes)}
                </dd>
              </div>
            </dl>
            <p>{plan.estimate_note}</p>
            {!!plan.conflict_count && (
              <p className="data-error-text">
                이미 존재하는 파일 {plan.conflict_count}개를 발견했습니다.
                보존을 위해 다른 폴더를 지정하세요.
              </p>
            )}
            {!plan.space_ok && (
              <p className="data-error-text">
                여유 공간이 부족합니다. 다른 드라이브 또는 더 작은 범위를
                선택하세요.
              </p>
            )}
            <div className="data-preview-actions">
              <a href={`/api/data/downloads/${plan.id}/manifest`} download>
                전체 파일 명세 JSON
              </a>
              <button
                className="button primary"
                disabled={disabled || !plan.space_ok || !!plan.conflict_count}
                onClick={() =>
                  run("다운로드를 시작하고 있습니다", async () => {
                    await api(`/data/downloads/${plan.id}/start`, post());
                    setPlan(null);
                    setJobs(await api("/data/downloads"));
                  })
                }
              >
                <ArrowDownToLine size={17} /> 다운로드 시작
              </button>
            </div>
          </div>
        )}
      </section>

      <section className="data-panel">
        <div className="data-section-heading">
          <div>
            <h2>
              <HardDrive size={20} /> 다운로드 이력
            </h2>
            <p>
              앱 실행 중에 전송됩니다. 중지된 파일은 보존하며, 서버가 지원하면
              이어받습니다.
            </p>
          </div>
        </div>
        {!jobs.length && (
          <div className="data-empty">
            아직 준비한 다운로드가 없습니다. 위에서 필요한 범위를 선택하세요.
          </div>
        )}
        <div className="data-job-list">
          {jobs
            .filter((j) => j.status !== "planned")
            .map((j) => (
              <article key={j.id}>
                <div className="data-job-main">
                  <div className="data-inline">
                    <strong>{modeNames[j.plan.mode]}</strong>
                    <span
                      className={
                        "badge " + (j.status === "complete" ? "good" : "")
                      }
                    >
                      {statusNames[j.status]}
                    </span>
                  </div>
                  <p>
                    {stamp(j.created_at)} KST ·{" "}
                    {j.progress.completed_files || 0}/{j.file_count} 파일 완료 ·
                    저장 {size(j.progress.completed_bytes || 0)}
                  </p>
                  <code className="data-path-text">{j.plan.destination}</code>
                  {["running", "queued"].includes(j.status) && (
                    <>
                      <progress
                        max={j.file_count}
                        value={
                          j.file_count > 1
                            ? j.progress.completed_files || 0
                            : undefined
                        }
                        aria-label="다운로드 진행"
                      />
                      <p className="data-path-text">
                        {j.progress.current_file || "연결 중"} · 현재 파일{" "}
                        {size(j.progress.current_bytes || 0)}
                      </p>
                    </>
                  )}
                  {archiveAvailable &&
                  j.plan.mode === "sample" &&
                  !["running", "queued", "stopping"].includes(j.status) ? (
                    <p className="data-hint">
                      이전 별도 다운로드 이력입니다. 공개 테스트는 연결된 전체
                      ZIP에 포함되어 있어 위 원본 목록에서 열 수 있습니다. 다시
                      받을 필요가 없습니다.
                    </p>
                  ) : (
                    j.error && <p className="data-error-text">{j.error}</p>
                  )}
                </div>
                <div className="data-job-actions">
                  <a
                    className="button secondary"
                    href={`/api/data/downloads/${j.id}/manifest`}
                    download
                  >
                    명세
                  </a>
                  {["running", "queued", "stopping"].includes(j.status) ? (
                    <button
                      className="button secondary"
                      disabled={!!busy}
                      onClick={() =>
                        run("다운로드를 중지하고 있습니다", async () => {
                          await api(`/data/downloads/${j.id}/cancel`, post());
                          setJobs(await api("/data/downloads"));
                        })
                      }
                    >
                      <Square size={14} /> 중지
                    </button>
                  ) : (
                    ["cancelled", "failed", "interrupted"].includes(j.status) &&
                    !(archiveAvailable && j.plan.mode === "sample") && (
                      <button
                        className="button secondary"
                        disabled={disabled}
                        onClick={() =>
                          run("다운로드를 다시 시작하고 있습니다", async () => {
                            await api(`/data/downloads/${j.id}/start`, post());
                            setJobs(await api("/data/downloads"));
                          })
                        }
                      >
                        재시도 / 이어받기
                      </button>
                    )
                  )}
                </div>
              </article>
            ))}
        </div>
      </section>

      <details className="data-panel data-details">
        <summary>대용량 데이터를 학습에 사용하는 권장 순서</summary>
        <ol className="data-strategy">
          <li>
            <strong>기존 캐시로 작은 기준 실험</strong>
            <p>
              Adjacent 캐시의 4,407 study를 memory map으로 읽습니다. 16 GB
              GPU에서는 작은 배치·혼합 정밀도로 시작하고 grouped fold, 라벨,
              결측 마스크를 고정하세요.
            </p>
          </li>
          <li>
            <strong>필요한 원본만 추가</strong>
            <p>
              슬라이스 범위·해상도·crop이 성능을 제한하는지 선택 study로
              확인합니다. DICOM 위치 정보로 정렬하고 추론에서도 같은 변환을
              사용하세요.
            </p>
          </li>
          <li>
            <strong>Kaggle에서 캐시를 만드는 방법</strong>
            <p>
              원본을 PC로 옮기지 않고 Kaggle Notebook의 대회 입력을 읽어 uint8
              NPY/NPZ 캐시를 생성할 수 있습니다. 계산 시간과 출력 저장 한도
              안에서 shard로 나눠 만들고 UID·변환·마스크 명세를 함께
              내려받으세요. 이 앱은 Notebook을 실행하지 않습니다.
            </p>
          </li>
          <li>
            <strong>전체 다운로드는 추가 실험의 필요성이 확인된 뒤</strong>
            <p>
              새로운 전체 볼륨 전처리·3D 실험처럼 기존 캐시로 할 수 없는 실험에
              사용하세요. ZIP을 풀 때는 원본 크기만큼의 추가 공간이 필요합니다.
            </p>
          </li>
        </ol>
        <p className="data-hint">
          Gold58은 이전 실험에서 사용되어 독립 holdout이 아닙니다. 테스트에는
          Report가 없으므로 추론 입력으로 사용하지 않습니다. 공개 노트북의
          점수는 검증된 입상 결과와 구분합니다.
        </p>
        <div className="data-source-links">
          <a
            href="https://www.kaggle.com/code/abisheksrivastav/rsna-knee-multi-plane-dinov2"
            target="_blank"
            rel="noreferrer"
          >
            참고: 슬롯당 12슬라이스·uint8 캐시 <ExternalLink size={14} />
          </a>
          <a
            href="https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee"
            target="_blank"
            rel="noreferrer"
          >
            참고: 메모리 예산 기반 슬라이스 수 <ExternalLink size={14} />
          </a>
        </div>
      </details>
    </div>
  );
}
