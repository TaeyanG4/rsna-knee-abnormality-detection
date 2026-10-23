import React, { useEffect, useState } from "react";
import { Folder, FileImage, Archive, Database } from "lucide-react";
type Row = Record<string, any>;
type Api = (path: string, options?: RequestInit) => Promise<any>;
type Open = (body: Row, focus?: Row) => void;

function FileBranch({
  api,
  study,
  series,
  onOpen,
  busy,
}: {
  api: Api;
  study: Row;
  series: Row;
  onOpen: Open;
  busy: boolean;
}) {
  const [expanded, setExpanded] = useState(false),
    [data, setData] = useState<Row | null>(null);
  const [offset, setOffset] = useState(0),
    [error, setError] = useState("");
  useEffect(() => {
    if (!expanded) return;
    let live = true;
    setError("");
    setData(null);
    api(
      `/data/structure?raw=${study.id}&series=${encodeURIComponent(series.series)}&offset=${offset}`,
    )
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [expanded, offset, study.id, series.series]);
  return (
    <details
      onToggle={(e) => {
        if (e.target === e.currentTarget) setExpanded(e.currentTarget.open);
      }}
      className="tree-series"
    >
      <summary title={series.series}>
        <Folder size={14} />
        <span>
          {series.description || series.series}
          <small>
            …{series.series.slice(-14)} · {series.files} DICOM
          </small>
        </span>
      </summary>
      <button
        className="tree-open"
        disabled={busy}
        onClick={() => onOpen({ raw: study.id }, { series_uid: series.series })}
      >
        시리즈 영상 열기
      </button>
      {expanded && !data && !error && <p role="status">파일 목록 읽는 중…</p>}
      {error && <p role="alert">{error}</p>}
      {data?.files.map((f: Row) => (
        <button
          className="tree-file"
          key={f.path}
          disabled={busy}
          title={f.path}
          onClick={() =>
            onOpen(
              { raw: study.id },
              { series_uid: series.series, filename: f.path.split("/").pop() },
            )
          }
        >
          <FileImage size={13} />
          <span>{f.path.split("/").pop()}</span>
        </button>
      ))}
      {data && data.total > 30 && (
        <div className="tree-paging">
          <button disabled={!offset} onClick={() => setOffset(offset - 30)}>
            이전
          </button>
          <span>
            {offset + 1}–{Math.min(offset + 30, data.total)}
          </span>
          <button
            disabled={offset + 30 >= data.total}
            onClick={() => setOffset(offset + 30)}
          >
            다음
          </button>
        </div>
      )}
    </details>
  );
}

function StudyBranch({
  api,
  study,
  source,
  onOpen,
  busy,
  selected,
}: {
  api: Api;
  study: Row;
  source: string;
  onOpen: Open;
  busy: boolean;
  selected: boolean;
}) {
  const [expanded, setExpanded] = useState(false),
    [data, setData] = useState<Row | null>(null),
    [error, setError] = useState("");
  const raw = source === "raw";
  useEffect(() => {
    if (!expanded || (raw && !study.archive_id)) return;
    let live = true;
    setError("");
    api(
      raw
        ? `/data/structure?raw=${study.id}`
        : `/viewer/tree/series?source=${source}&uid=${encodeURIComponent(study.uid)}`,
    )
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [expanded, source, study.id, study.uid]);
  return (
    <details
      className={"tree-study " + (selected ? "selected" : "")}
      onToggle={(e) => {
        if (e.target === e.currentTarget) setExpanded(e.currentTarget.open);
      }}
    >
      <summary title={study.uid}>
        <Folder size={14} />
        <span>
          …{study.uid.slice(-18)}
          <small>
            {raw
              ? `${study.files}장 · ${study.status === "archived" ? "ZIP" : "로컬"}`
              : "캐시 study"}
            {study.labeled ? " · 라벨" : ""}
            {study.missing ? ` · 결측 ${study.missing}` : ""}
          </small>
        </span>
      </summary>
      <button
        className="tree-open"
        disabled={busy}
        onClick={() =>
          onOpen(raw ? { raw: study.id } : { source, uid: study.uid })
        }
      >
        study 영상 열기
      </button>
      {error && <p role="alert">{error}</p>}
      {data?.series.map((s: Row) =>
        raw ? (
          <FileBranch
            key={s.series}
            api={api}
            study={study}
            series={s}
            onOpen={onOpen}
            busy={busy}
          />
        ) : (
          <details className="tree-series" key={s.slot}>
            <summary>
              <Folder size={14} />
              <span>
                {s.name}
                <small>
                  {s.slices}장 · 유효 {s.valid.filter(Boolean).length}
                </small>
              </span>
            </summary>
            {s.valid.map((valid: boolean, z: number) => (
              <button
                key={z}
                disabled={busy}
                className="tree-file"
                onClick={() =>
                  onOpen({ source, uid: study.uid }, { slot: s.slot, slice: z })
                }
              >
                <FileImage size={13} />
                슬라이스 {z + 1}
                {valid ? "" : " · 결측"}
              </button>
            ))}
          </details>
        ),
      )}
    </details>
  );
}

function StudyFolder({
  api,
  source,
  root = "",
  split = "",
  label,
  query,
  mode,
  target,
  onOpen,
  busy,
  session,
}: Row) {
  const [expanded, setExpanded] = useState(false),
    [offset, setOffset] = useState(0),
    [data, setData] = useState<Row | null>(null),
    [error, setError] = useState("");
  useEffect(() => setOffset(0), [query, mode, target]);
  useEffect(() => {
    if (!expanded) return;
    let live = true;
    setData(null);
    setError("");
    api(
      `/viewer/studies?source=${source}&root=${encodeURIComponent(root)}&split=${split}&q=${encodeURIComponent(query)}&mode=${mode}&target=${encodeURIComponent(target)}&offset=${offset}`,
    )
      .then((d: Row) => live && setData(d))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [expanded, query, mode, target, offset, source, root, split]);
  return (
    <details
      className="tree-folder"
      onToggle={(e) => {
        if (e.target === e.currentTarget) setExpanded(e.currentTarget.open);
      }}
    >
      <summary>
        <Folder size={16} />
        <span>{label}</span>
      </summary>
      {error && <p role="alert">{error}</p>}
      {expanded && !data && !error && <p role="status">study 목록 읽는 중…</p>}
      {data && (
        <>
          <p className="tree-caption">
            검색 {data.total.toLocaleString()}건 · 40건씩
          </p>
          {data.items.map((s: Row) => (
            <StudyBranch
              key={s.id || s.uid}
              api={api}
              source={source}
              study={s}
              onOpen={onOpen}
              busy={busy}
              selected={
                source === "raw"
                  ? session?.raw_id === s.id
                  : session?.sourceId === source && session?.uid === s.uid
              }
            />
          ))}
          {data.total > 40 && (
            <div className="tree-paging">
              <button disabled={!offset} onClick={() => setOffset(offset - 40)}>
                이전 study
              </button>
              <span>
                {offset + 1}–{Math.min(offset + 40, data.total)}
              </span>
              <button
                disabled={offset + 40 >= data.total}
                onClick={() => setOffset(offset + 40)}
              >
                다음 study
              </button>
            </div>
          )}
          {!data.total && <p>조건에 맞는 study가 없습니다.</p>}
        </>
      )}
    </details>
  );
}

function LocalFolder({ api, root, relative = "", label, onOpen, busy }: Row) {
  const [expanded, setExpanded] = useState(false),
    [data, setData] = useState<Row | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    if (!expanded) return;
    let live = true;
    api(`/viewer/browse?root=${root}&relative=${encodeURIComponent(relative)}`)
      .then((d: Row) => live && setData(d))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [expanded, root, relative]);
  return (
    <details
      className="tree-folder"
      onToggle={(e) => {
        if (e.target === e.currentTarget) setExpanded(e.currentTarget.open);
      }}
    >
      <summary>
        <Folder size={14} />
        <span>{label}</span>
      </summary>
      {error && <p role="alert">{error}</p>}
      {data && (
        <>
          {relative && (
            <button
              className="tree-open"
              disabled={busy}
              onClick={() => onOpen({ path: data.path })}
            >
              이 study/시리즈 폴더 열기
            </button>
          )}
          {data.items.map((r: Row) =>
            r.directory ? (
              <LocalFolder
                key={r.relative}
                api={api}
                root={root}
                relative={r.relative}
                label={r.name}
                onOpen={onOpen}
                busy={busy}
              />
            ) : (
              <button
                key={r.relative}
                className="tree-file"
                disabled={busy}
                onClick={() => onOpen({ path: r.path })}
                title={r.name}
              >
                <FileImage size={13} />
                <span>{r.name}</span>
              </button>
            ),
          )}
          {data.truncated && (
            <p>
              큰 폴더는 일부만 표시합니다. 전체 원본은 위 ZIP 목록을 사용하세요.
            </p>
          )}
        </>
      )}
    </details>
  );
}

export default function ViewerTree({
  api,
  catalog,
  query,
  mode,
  target,
  onOpen,
  busy,
  session,
}: Row) {
  const groups = new Map<string, string>();
  for (const row of catalog.raw || [])
    groups.set(
      row.archive_id || row.root,
      row.archive_id ? "전체 원본 ZIP" : row.root,
    );
  const props = { api, query, mode, target, onOpen, busy, session };
  return (
    <div className="viewer-folder-tree" aria-label="데이터 폴더 탐색">
      <p className="tree-caption">
        폴더를 펼쳐 study → 시리즈 → 슬라이스를 선택하세요. ZIP 탐색은 영상을
        해제하지 않습니다.
      </p>
      {[...groups].map(([key, label]) => (
        <details key={key} open className="tree-root">
          <summary>
            <Archive size={17} />
            {label}
          </summary>
          <StudyFolder
            {...props}
            source="raw"
            root={key}
            split="train"
            label="train_series · 학습 원본"
          />
          <StudyFolder
            {...props}
            source="raw"
            root={key}
            split="test"
            label="test_series · 공개 테스트"
          />
        </details>
      ))}
      <details open className="tree-root">
        <summary>
          <Database size={17} />
          전처리 캐시
        </summary>
        {catalog.caches.map((c: Row) => (
          <StudyFolder key={c.id} {...props} source={c.id} label={c.name} />
        ))}
      </details>
      <details className="tree-root">
        <summary>
          <Folder size={17} />
          실제 로컬 폴더
        </summary>
        {catalog.roots.map((r: Row) => (
          <LocalFolder
            key={r.id}
            api={api}
            root={r.id}
            label={r.name}
            onOpen={onOpen}
            busy={busy}
          />
        ))}
      </details>
    </div>
  );
}
