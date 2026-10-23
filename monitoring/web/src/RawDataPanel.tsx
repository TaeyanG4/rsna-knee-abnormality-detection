import React, { useEffect, useState } from "react";
import { ArrowDownToLine, FolderOpen, RefreshCw } from "lucide-react";
import { DatasetDescription, StudyStructure, bytes } from "./DataStructure";
type Row = Record<string, any>;
type Api = (path: string, options?: RequestInit) => Promise<any>;
const states: Record<string, string> = {
  complete: "목록·크기 일치",
  partial: "일부만 있음",
  missing: "미다운로드",
  unverified: "완전성 미확인",
  archived: "ZIP 보관 · 열 때 준비",
};

export default function RawDataPanel({ api }: { api: Api }) {
  const [data, setData] = useState<Row | null>(null),
    [job, setJob] = useState(""),
    [error, setError] = useState("");
  const [query, setQuery] = useState(""),
    [split, setSplit] = useState("all"),
    [offset, setOffset] = useState(0);
  const filtered = (data?.studies || []).filter(
    (s: Row) => s.uid.includes(query) && (split === "all" || s.split === split),
  );
  useEffect(() => {
    let live = true;
    api("/data/library")
      .then((value) => live && setData(value))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    if (!job) return;
    let live = true;
    const poll = async () => {
      try {
        const jobs = await api("/jobs");
        const current = jobs.find((x: Row) => x.id === job);
        if (!live || !current || ["queued", "running"].includes(current.status))
          return;
        if (current.status !== "complete")
          setError(current.error || "로컬 목록 확인이 중단되었습니다.");
        const value = await api("/data/library");
        if (live) {
          setData(value);
          setJob("");
        }
      } catch (e) {
        if (live) {
          setError((e as Error).message);
          setJob("");
        }
      }
    };
    const timer = setInterval(poll, 1000);
    poll();
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [job]);
  const index = async () => {
    setError("");
    try {
      const value = await api("/jobs", {
        method: "POST",
        body: JSON.stringify({ kind: "data_index" }),
      });
      setJob(value.id);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  return (
    <section className="data-panel raw-data-panel">
      <div className="data-section-heading">
        <div>
          <h2>
            <FolderOpen size={20} />
            로컬 원본 데이터
          </h2>
          <p>
            전체 ZIP과 준비된 DICOM을 검색하고 원본 해상도로 엽니다. ZIP은
            선택한 study의 모든 시리즈만 해제해 보관하며, 다음에 열 때
            재사용합니다.
          </p>
        </div>
        <button className="button secondary" disabled={!!job} onClick={index}>
          <RefreshCw size={16} className={job ? "spin" : ""} />
          {job ? "목록 확인 중" : "로컬 목록 새로고침"}
        </button>
      </div>
      {error && (
        <p role="alert" className="data-error-text">
          {error}
        </p>
      )}
      {data?.indexed_at ? (
        <>
          {!!data.archives?.length && (
            <>
              <div className="raw-archive-summary">
                <strong>
                  전체 원본 연결됨 · {data.studies.length.toLocaleString()}{" "}
                  study
                </strong>
                <p>
                  ZIP{" "}
                  {bytes(
                    data.archives.reduce(
                      (n: number, a: Row) => n + a.zip_bytes,
                      0,
                    ),
                  )}{" "}
                  · 모두 해제할 경우{" "}
                  {bytes(
                    data.archives.reduce((n: number, a: Row) => n + a.bytes, 0),
                  )}
                </p>
                <p>
                  현재 준비된 study {data.prepared_studies || 0}건 ·{" "}
                  {bytes(data.prepared_bytes || 0)}. 전체 영상 CRC 검사는 아직
                  하지 않았으며, 각 study를 준비할 때 CRC와 SHA256을 확인합니다.
                </p>
                {data.archives.some((a: Row) => !a.available) && (
                  <p role="alert">
                    원본 ZIP이 이동·변경되었습니다. 경로를 확인하고 목록을
                    새로고침하세요.
                  </p>
                )}
              </div>
              <DatasetDescription data={data} />
            </>
          )}
          <p className="data-hint">
            확인 {new Date(data.indexed_at).toLocaleString("ko-KR")} ·{" "}
            {data.archives?.length ? "ZIP 포함 " : ""}DICOM{" "}
            {data.files.toLocaleString()}개 · 학습 캐시{" "}
            {data.cache_studies.toLocaleString()} study
          </p>
          <div className="raw-filters">
            <label>
              Study UID 검색
              <input
                aria-label="원본 Study UID 검색"
                value={query}
                placeholder="UID 일부로 검색"
                onChange={(e) => {
                  setQuery(e.target.value);
                  setOffset(0);
                }}
              />
            </label>
            <label>
              데이터 구분
              <select
                aria-label="원본 데이터 구분"
                value={split}
                onChange={(e) => {
                  setSplit(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="all">전체</option>
                <option value="train">학습 원본</option>
                <option value="test">공개 테스트</option>
              </select>
            </label>
            <p>{filtered.length.toLocaleString()}건 · 한 페이지 20건</p>
          </div>
          <div className="raw-study-list">
            {filtered.slice(offset, offset + 20).map((study: Row) => (
              <article key={study.id}>
                <div>
                  <div className="data-inline">
                    <strong>
                      {study.split === "test"
                        ? "공개 테스트 · 추론 점검용"
                        : "학습 원본"}
                    </strong>
                    <span className="badge">
                      {study.storage === "extracted"
                        ? "원본 준비됨 · CRC 확인"
                        : states[study.status]}
                    </span>
                  </div>
                  <code>{study.uid}</code>
                  <p>
                    파일 {study.files.toLocaleString()}
                    {study.expected_files != null &&
                      ` / ${study.expected_files.toLocaleString()}`}{" "}
                    · 시리즈 {study.series}
                    {study.expected_series != null &&
                      ` / ${study.expected_series}`}
                  </p>
                  <details>
                    <summary>
                      {study.status === "archived"
                        ? "열 때 저장될 위치"
                        : "파일 위치"}
                    </summary>
                    <code>{study.path}</code>
                  </details>
                  {study.archive_id && (
                    <StudyStructure api={api} id={study.id} />
                  )}
                </div>
                {study.files > 0 ? (
                  <a
                    className="button secondary"
                    href={`/?raw=${study.id}#viewer`}
                  >
                    <FolderOpen size={16} />
                    {study.status === "archived"
                      ? "원본 준비 후 열기"
                      : "영상 열기"}
                  </a>
                ) : (
                  <span className="data-hint">다운로드 필요</span>
                )}
              </article>
            ))}
          </div>
          <div className="raw-pagination">
            <button
              className="button secondary"
              disabled={!offset}
              onClick={() => setOffset(offset - 20)}
            >
              이전 study
            </button>
            <span>
              {filtered.length ? offset + 1 : 0}–
              {Math.min(offset + 20, filtered.length)} /{" "}
              {filtered.length.toLocaleString()}건
            </span>
            <button
              className="button secondary"
              disabled={offset + 20 >= filtered.length}
              onClick={() => setOffset(offset + 20)}
            >
              다음 study
            </button>
          </div>
          {!data.studies.length && (
            <p className="data-hint">
              아직 풀린 원본이 없습니다. 캐시는 영상 분석에서 바로 열 수
              있습니다.
            </p>
          )}
          <div className="data-cache-footer">
            <p>
              공개 테스트에는 학습 정답·판독문이 없습니다. 일부만 받은 study는
              학습 입력으로 사용하지 않습니다.
            </p>
            <a
              className="button secondary"
              href="/api/data/library/descriptor"
              download
            >
              <ArrowDownToLine size={16} />
              로컬 데이터 명세 JSON
            </a>
          </div>
        </>
      ) : (
        <p className="data-hint">
          목록을 새로고침하면 원본 위치·파일 수·누락 여부와 학습용 캐시를 함께
          정리합니다.
        </p>
      )}
    </section>
  );
}
