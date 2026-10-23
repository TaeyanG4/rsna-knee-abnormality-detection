import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ChevronLeft,
  ChevronRight,
  FolderOpen,
  Hand,
  LoaderCircle,
  Maximize,
  Pause,
  Play,
  RefreshCw,
  Ruler,
  Scan,
  Search,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import "./viewer.css";
import { ViewerFrameCache } from "./viewer-frame-cache";

type Row = Record<string, any>;
type Api = (path: string, options?: RequestInit) => Promise<any>;
type Box = [number, number, number, number];
const post = (body: Row) => ({ method: "POST", body: JSON.stringify(body) });
const fmt = (n: number | null | undefined) =>
  n == null ? "—" : n.toLocaleString("ko-KR", { maximumFractionDigits: 2 });
const flags: Record<string, string> = {
  checked: "확인 완료",
  motion: "움직임·아티팩트",
  crop: "잘림 확인",
  missing: "결측 확인",
  label_check: "라벨 재검토",
};
const targetNames: Record<string, string> = {
  ACL: "전방십자인대",
  MCL: "내측측부인대",
  "Medial Meniscus": "내측 반월상연골",
  "Lateral Meniscus": "외측 반월상연골",
  "Medial OA": "내측 골관절염",
  "Lateral OA": "외측 골관절염",
  "PF OA": "슬개대퇴 골관절염",
  Effusion: "관절삼출",
  Synovitis: "활막염",
  "Baker's": "베이커 낭종",
  Contusion: "골좌상",
  Fracture: "골절",
};

function ImageCanvas({
  frame,
  roi,
  onRoi,
  onStep,
  busy,
}: {
  frame: Row | null;
  roi: Box | null;
  onRoi: (box: Box | null) => void;
  onStep: (delta: number) => void;
  busy: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement>(null),
    container = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState([700, 470]);
  const picture = frame?.decodedImage as HTMLImageElement | undefined;
  const [slow, setSlow] = useState(false);
  const [tool, setTool] = useState("pan"),
    [zoom, setZoom] = useState(1),
    [pan, setPan] = useState([0, 0]);
  const [draft, setDraft] = useState<Box | null>(null),
    [line, setLine] = useState<Box | null>(null);
  const drag = useRef<{
    start: number[];
    pan: number[];
    image: number[];
  } | null>(null);
  const stepRef = useRef(onStep);
  stepRef.current = onStep;
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setSize([entry.contentRect.width, entry.contentRect.height]),
    );
    if (container.current) observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    setSlow(false);
    if (!busy) return;
    const timer = setTimeout(() => setSlow(true), 700);
    return () => clearTimeout(timer);
  }, [busy]);
  useEffect(() => {
    setLine(null);
    setDraft(null);
    drag.current = null;
  }, [frame?.slot, frame?.slice]);
  const shape = frame?.shape || [1, 1];
  const aspect = frame?.spacing ? frame.spacing[1] / frame.spacing[0] : 1;
  const scale =
    Math.min((size[0] - 32) / (shape[1] * aspect), (size[1] - 32) / shape[0]) *
    zoom;
  const width = shape[1] * aspect * scale,
    height = shape[0] * scale;
  const left = (size[0] - width) / 2 + pan[0],
    top = (size[1] - height) / 2 + pan[1];
  const reset = () => {
    setZoom(1);
    setPan([0, 0]);
  };
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey)
        setZoom((z) =>
          Math.max(0.5, Math.min(8, z + (e.deltaY < 0 ? 0.25 : -0.25))),
        );
      else stepRef.current(e.deltaY > 0 ? 1 : -1);
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, []);
  useLayoutEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = size[0] * dpr;
    el.height = size[1] * dpr;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#090f15";
    ctx.fillRect(0, 0, ...(size as [number, number]));
    if (picture) ctx.drawImage(picture, left, top, width, height);
    const box = draft || roi;
    if (box) {
      ctx.strokeStyle = "#5eead4";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(
        left + box[0] * width,
        top + box[1] * height,
        (box[2] - box[0]) * width,
        (box[3] - box[1]) * height,
      );
      ctx.setLineDash([]);
    }
    if (line) {
      ctx.strokeStyle = "#fcd34d";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(left + line[0] * width, top + line[1] * height);
      ctx.lineTo(left + line[2] * width, top + line[3] * height);
      ctx.stroke();
    }
  }, [picture, size, zoom, pan, roi, draft, line, frame]);
  const coords = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const raw = [e.clientX - rect.left, e.clientY - rect.top];
    return {
      raw,
      image: [
        Math.max(0, Math.min(1, (raw[0] - left) / width)),
        Math.max(0, Math.min(1, (raw[1] - top) / height)),
      ],
    };
  };
  const distance = line
    ? Math.hypot(
        (line[2] - line[0]) * shape[1] * (frame?.spacing?.[1] || 1),
        (line[3] - line[1]) * shape[0] * (frame?.spacing?.[0] || 1),
      )
    : null;
  return (
    <div className="viewer-image-area">
      <div className="viewer-tools" role="toolbar" aria-label="영상 도구">
        {[
          ["pan", Hand, "이동"],
          ["roi", Scan, "영역 선택"],
          ["ruler", Ruler, "길이"],
        ].map(([id, Icon, label]) => {
          const ToolIcon = Icon as typeof Hand;
          return (
            <button
              key={String(id)}
              aria-pressed={tool === id}
              onClick={() => setTool(String(id))}
            >
              <ToolIcon size={17} />
              {String(label)}
            </button>
          );
        })}
        <span className="viewer-tool-divider" />
        <button
          aria-label="축소"
          onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
        >
          <ZoomOut size={18} />
        </button>
        <span className="viewer-zoom">{Math.round(zoom * 100)}%</span>
        <button
          aria-label="확대"
          onClick={() => setZoom((z) => Math.min(8, z + 0.25))}
        >
          <ZoomIn size={18} />
        </button>
        <button onClick={reset}>
          <Maximize size={17} />
          맞춤
        </button>
      </div>
      <div className="viewer-stage" ref={container} aria-busy={busy}>
        <canvas
          ref={canvas}
          tabIndex={0}
          aria-label="MRI 영상 캔버스"
          style={{ cursor: tool === "pan" ? "grab" : "crosshair" }}
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
              e.preventDefault();
              onStep(e.key === "ArrowRight" ? 1 : -1);
            }
            if (e.key === "Escape") {
              onRoi(null);
              setLine(null);
            }
          }}
          onPointerDown={(e) => {
            if (!frame || busy) return;
            e.currentTarget.focus();
            e.currentTarget.setPointerCapture(e.pointerId);
            const c = coords(e);
            drag.current = { start: c.raw, pan, image: c.image };
          }}
          onPointerMove={(e) => {
            const start = drag.current;
            if (!start) return;
            const c = coords(e);
            if (tool === "pan")
              setPan([
                start.pan[0] + c.raw[0] - start.start[0],
                start.pan[1] + c.raw[1] - start.start[1],
              ]);
            else if (tool === "roi")
              setDraft([
                Math.min(start.image[0], c.image[0]),
                Math.min(start.image[1], c.image[1]),
                Math.max(start.image[0], c.image[0]),
                Math.max(start.image[1], c.image[1]),
              ]);
            else setLine([...start.image, ...c.image] as Box);
          }}
          onPointerUp={(e) => {
            if (drag.current && tool === "roi") {
              const start = drag.current.image,
                end = coords(e).image;
              const box: Box = [
                Math.min(start[0], end[0]),
                Math.min(start[1], end[1]),
                Math.max(start[0], end[0]),
                Math.max(start[1], end[1]),
              ];
              if (box[2] - box[0] > 0.001 && box[3] - box[1] > 0.001)
                onRoi(box);
            }
            drag.current = null;
            setDraft(null);
          }}
          onPointerCancel={() => {
            drag.current = null;
            setDraft(null);
          }}
        />
        <div className="viewer-stage-label">
          {frame
            ? `${shape[1]} × ${shape[0]} · ${frame.intensity_unit} · #${frame.slice + 1}`
            : "MRI VIEWER"}
        </div>
        {distance != null && (
          <div className="viewer-measure">
            {fmt(distance)} {frame?.spacing ? "mm" : "px"}{" "}
            <button onClick={() => setLine(null)} aria-label="길이 측정 지우기">
              ×
            </button>
          </div>
        )}
        {busy && !frame && (
          <div className="viewer-loading">
            <LoaderCircle className="spin" size={22} />
            <span>슬라이스 읽는 중</span>
          </div>
        )}
        {busy && frame && slow && (
          <div className="viewer-buffering" role="status">
            다음 영상 준비 중 · #{frame.slice + 1} 유지
          </div>
        )}
        {frame && !frame.valid && (
          <div className="viewer-missing">
            결측 마스크 · 유효 영상으로 해석하지 마세요
          </div>
        )}
      </div>
      <p className="viewer-shortcuts">
        휠·← →: 슬라이스 이동 · Ctrl+휠: 확대 · 드래그: 선택 도구 · Esc: 선택
        해제
      </p>
    </div>
  );
}

function Histogram({ info }: { info: Row | undefined }) {
  if (!info?.histogram?.length)
    return <p className="viewer-muted">표시할 픽셀 통계가 없습니다.</p>;
  const maximum = Math.max(...info.histogram, 1);
  return (
    <>
      <svg
        className="viewer-histogram"
        viewBox="0 0 320 90"
        role="img"
        aria-label="픽셀 강도 히스토그램"
      >
        <line
          x1="0"
          y1="88"
          x2="320"
          y2="88"
          stroke="currentColor"
          opacity=".2"
        />
        {info.histogram.map((n: number, i: number) => (
          <rect
            key={i}
            x={i * 10}
            y={88 - (n / maximum) * 80}
            width="8"
            height={(n / maximum) * 80}
            fill="currentColor"
          >
            <title>
              {fmt(info.bin_edges[i])}–{fmt(info.bin_edges[i + 1])}: {n} pixels
            </title>
          </rect>
        ))}
      </svg>
      <div className="viewer-histogram-range">
        <span>{fmt(info.min)}</span>
        <span>{fmt(info.max)}</span>
      </div>
      <dl className="viewer-stats">
        <div>
          <dt>평균</dt>
          <dd>{fmt(info.mean)}</dd>
        </div>
        <div>
          <dt>표준편차</dt>
          <dd>{fmt(info.std)}</dd>
        </div>
        <div>
          <dt>픽셀 수</dt>
          <dd>{fmt(info.pixels)}</dd>
        </div>
        <div>
          <dt>0 값 비율</dt>
          <dd>{fmt(info.zero_percent)}%</dd>
        </div>
      </dl>
    </>
  );
}

import ViewerTree from "./ViewerTree";
import { DatasetEda, SeriesEda, PreprocessingEda } from "./ViewerEda";
import ViewerCompare from "./ViewerCompare";
import "./viewer-eda.css";

export default function ViewerPage({ api }: { api: Api }) {
  const [navigation, setNavigation] = useState("tree");
  const [catalog, setCatalog] = useState<Row>({ caches: [], roots: [] }),
    [source, setSource] = useState("");
  const [query, setQuery] = useState(""),
    [mode, setMode] = useState("all"),
    [target, setTarget] = useState(""),
    [offset, setOffset] = useState(0);
  const [studies, setStudies] = useState<Row>({ items: [], total: 0 }),
    [listBusy, setListBusy] = useState(false);
  const [session, setSession] = useState<Row | null>(null),
    [slot, setSlot] = useState(0),
    [slice, setSlice] = useState(0);
  const [frame, setFrame] = useState<Row | null>(null),
    [windowRange, setWindowRange] = useState<[number, number] | null>(null),
    [invert, setInvert] = useState(false),
    [roi, setRoi] = useState<Box | null>(null);
  const [busy, setBusy] = useState(false),
    [frameBusy, setFrameBusy] = useState(false),
    [error, setError] = useState(""),
    [playing, setPlaying] = useState(false);
  const [path, setPath] = useState(""),
    [root, setRoot] = useState(""),
    [folder, setFolder] = useState<Row | null>(null);
  const [contact, setContact] = useState<Row | null>(null),
    [contactBusy, setContactBusy] = useState(false);
  const [note, setNote] = useState(""),
    [selectedFlags, setSelectedFlags] = useState<string[]>([]),
    [saved, setSaved] = useState(""),
    [saving, setSaving] = useState(false),
    [reviews, setReviews] = useState<Row[]>([]);
  const frameCache = useRef(new ViewerFrameCache());
  const readyRef = useRef(false);
  const selectionRef = useRef("");
  selectionRef.current = `${session?.id}:${slot}:${slice}`;
  const openedBody = useRef("");
  const openSeq = useRef(0),
    contactSeq = useRef(0),
    browseSeq = useRef(0);
  const series = session?.series[slot];
  const desiredFrameBody = {
    session: session?.id,
    slot,
    slice,
    low: windowRange?.[0] ?? null,
    high: windowRange?.[1] ?? null,
    invert,
    roi,
  };
  const current =
    frame &&
    frame.requestKey === JSON.stringify(desiredFrameBody) &&
    !frameBusy &&
    !busy;
  readyRef.current = Boolean(current);
  const frameBody = () => desiredFrameBody;
  useEffect(() => {
    let live = true;
    api("/viewer/catalog")
      .then((data) => {
        if (live) {
          setCatalog(data);
          const rawId = new URLSearchParams(location.search).get("raw");
          if (
            rawId &&
            data.raw?.some((row: Row) => row.id === rawId && row.files)
          ) {
            setSource("raw");
            open({ raw: rawId });
            return;
          }
          setSource(
            data.caches.find((x: Row) => x.selected)?.id ||
              data.caches[0]?.id ||
              (data.raw?.some((row: Row) => row.files) ? "raw" : ""),
          );
        }
      })
      .catch((e) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    if (!source) return;
    let live = true;
    setListBusy(true);
    const timer = setTimeout(
      () =>
        api(
          `/viewer/studies?source=${encodeURIComponent(source)}&q=${encodeURIComponent(query)}&mode=${mode}&target=${encodeURIComponent(target)}&offset=${offset}`,
        )
          .then((data) => live && setStudies(data))
          .catch((e) => live && setError(e.message))
          .finally(() => live && setListBusy(false)),
      180,
    );
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [source, query, mode, target, offset]);
  useEffect(() => {
    const cache = frameCache.current;
    if (!session) {
      cache.reset();
      return;
    }
    let live = true;
    const body = frameBody();
    cache.configure(body);
    const cached = cache.peek(body);
    setFrameBusy(!cached);
    setError("");
    if (cached) setFrame(cached);
    const load = async () => {
      try {
        const data = cached || (await cache.get(api, body));
        if (!live) return;
        setFrame(data);
        setFrameBusy(false);
        // One bounded prefetch chain; obsolete chains stop after their current request.
        for (const delta of [1, -1, 2, 3, -2, 4]) {
          if (!live) return;
          const z = (slice + delta + series.slices) % series.slices;
          try {
            await cache.get(api, { ...body, slice: z });
          } catch {
            break;
          }
        }
      } catch (e) {
        if (live && (e as Error).name !== "AbortError") {
          cache.reset();
          setError((e as Error).message);
          setFrame(null);
          setFrameBusy(false);
          setPlaying(false);
        }
      }
    };
    // Coalesce rapid wheel events; cached frames need no request or artificial delay.
    const timer = setTimeout(load, cached ? 0 : 16);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [session?.id, slot, slice, windowRange, invert, roi]);
  useEffect(() => () => frameCache.current.reset(), []);
  useEffect(() => {
    if (!playing || !series || busy) return;
    const timer = setInterval(() => {
      if (readyRef.current) setSlice((z) => (z + 1) % series.slices);
    }, 250);
    return () => clearInterval(timer);
  }, [playing, series, busy]);
  useEffect(() => {
    setSaved("");
  }, [session?.id, slot, slice]);
  useEffect(() => {
    const prior = reviews.find(
      (r) =>
        r.revision === session?.revision &&
        r.slot === slot &&
        r.slice === slice,
    );
    setNote(prior?.note || "");
    setSelectedFlags(prior?.flags || []);
  }, [session?.id, slot, slice, reviews]);
  useEffect(
    () => () => {
      ++openSeq.current;
      ++browseSeq.current;
      ++contactSeq.current;
    },
    [],
  );
  const focusedPosition = (data: Row, focus: Row) => {
    const requested = focus.series_uid
      ? data.series.findIndex((s: Row) => s.series_uid === focus.series_uid)
      : (focus.slot ?? 0);
    const selectedSlot = Math.max(
      0,
      Math.min(requested, data.series.length - 1),
    );
    const selectedSeries = data.series[selectedSlot];
    const fileSlice = focus.filename
      ? selectedSeries.filenames?.indexOf(focus.filename)
      : undefined;
    const selectedSlice =
      fileSlice != null && fileSlice >= 0
        ? fileSlice
        : (focus.slice ?? Math.floor(selectedSeries.slices / 2));
    return [
      selectedSlot,
      Math.max(0, Math.min(selectedSlice, selectedSeries.slices - 1)),
    ];
  };
  const open = async (body: Row, focus: Row = {}) => {
    if (
      session &&
      Object.keys(focus).length &&
      openedBody.current === JSON.stringify(body)
    ) {
      const [nextSlot, nextSlice] = focusedPosition(session, focus);
      setPlaying(false);
      if (nextSlot !== slot) chooseSeries(nextSlot);
      setSlice(nextSlice);
      return;
    }
    const seq = ++openSeq.current;
    ++contactSeq.current;
    setBusy(true);
    setPlaying(false);
    setError("");
    setContact(null);
    setContactBusy(false);
    setSession(null);
    setFrame(null);
    try {
      const data = await api("/viewer/open", post(body));
      if (seq !== openSeq.current) return;
      setSession({ ...data, sourceId: body.source });
      openedBody.current = JSON.stringify(body);
      const [selectedSlot, selectedSlice] = focusedPosition(data, focus);
      setSlot(selectedSlot);
      setSlice(selectedSlice);
      setWindowRange(null);
      setInvert(false);
      setRoi(null);
      setReviews(data.reviews || []);
    } catch (e) {
      if (seq === openSeq.current) setError((e as Error).message);
    } finally {
      if (seq === openSeq.current) setBusy(false);
    }
  };
  const browse = async (rootId: string, relative = "") => {
    const seq = ++browseSeq.current;
    setRoot(rootId);
    setError("");
    if (!rootId) {
      setFolder(null);
      return;
    }
    try {
      const data = await api(
        `/viewer/browse?root=${rootId}&relative=${encodeURIComponent(relative)}`,
      );
      if (seq === browseSeq.current) setFolder(data);
    } catch (e) {
      if (seq === browseSeq.current) setError((e as Error).message);
    }
  };
  const step = (delta: number) => {
    setPlaying(false);
    setSlice((z) =>
      Math.max(0, Math.min((series?.slices || 1) - 1, z + delta)),
    );
  };
  const chooseSeries = (i: number) => {
    ++contactSeq.current;
    setPlaying(false);
    setSlot(i);
    setFrame(null);
    setSlice(Math.floor(session!.series[i].slices / 2));
    setRoi(null);
    setWindowRange(null);
    setContact(null);
    setContactBusy(false);
  };
  const showContact = async () => {
    const seq = ++contactSeq.current;
    setContactBusy(true);
    setPlaying(false);
    try {
      const data = await api(
        `/viewer/contact?session=${session!.id}&slot=${slot}`,
      );
      if (seq === contactSeq.current) setContact(data);
    } catch (e) {
      if (seq === contactSeq.current) setError((e as Error).message);
    } finally {
      if (seq === contactSeq.current) setContactBusy(false);
    }
  };
  const save = async () => {
    const seq = openSeq.current,
      selection = selectionRef.current;
    setSaved("");
    setSaving(true);
    setPlaying(false);
    try {
      const data = await api(
        "/viewer/reviews",
        post({ ...frameBody(), note, flags: selectedFlags }),
      );
      if (seq === openSeq.current) {
        setReviews((list) => [data, ...list.filter((x) => x.id !== data.id)]);
        if (selection === selectionRef.current)
          setSaved("이 슬라이스의 검토 기록을 저장했습니다.");
      }
    } catch (e) {
      if (seq === openSeq.current) setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };
  return (
    <div className="viewer-page">
      <div className="viewer-intro">
        <div>
          <span className="eyebrow">LOCAL MRI WORKBENCH</span>
          <h2>
            <Activity size={24} />
            영상과 라벨을 한곳에서
          </h2>
          <p>
            캐시는 필요한 슬라이스만 읽습니다. 원본 파일은 변경하지 않습니다.
          </p>
        </div>
        <a className="button secondary" href="#data">
          다운로드·캐시 관리
        </a>
      </div>
      <details className="viewer-file-panel">
        <summary>
          <FolderOpen size={19} /> 로컬 파일·폴더 열기{" "}
          <span>DICOM · NPY · 캐시 manifest</span>
        </summary>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            open({ path: path.trim() });
          }}
        >
          <label htmlFor="viewer-path">이 PC의 파일 또는 study 폴더 경로</label>
          <div className="viewer-path-row">
            <input
              id="viewer-path"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="예: H:/.../train_series/StudyUID 또는 cache_manifest.json"
            />
            <button className="button primary" disabled={busy || !path.trim()}>
              열기
            </button>
          </div>
        </form>
        <p className="viewer-muted">
          DICOM은 한 study 폴더를 선택하세요. NPY는 [H,W]·[Z,H,W], 5차원 shard는
          manifest로 엽니다. 압축 ZIP은 먼저 풀어 주세요.
        </p>
        <label>
          프로젝트 데이터 탐색
          <select
            aria-label="탐색할 데이터 폴더"
            value={root}
            onChange={(e) => browse(e.target.value)}
          >
            <option value="">폴더 선택</option>
            {catalog.roots.map((r: Row) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
        </label>
        {folder && (
          <div className="viewer-file-browser">
            <div>
              <code>{folder.path}</code>
              <button
                className="button secondary"
                onClick={() => {
                  setPath(folder.path);
                  open({ path: folder.path });
                }}
              >
                이 폴더 열기
              </button>
              {folder.relative !== "." && (
                <button
                  className="button secondary"
                  onClick={() =>
                    browse(
                      root,
                      folder.relative.split("/").slice(0, -1).join("/"),
                    )
                  }
                >
                  상위 폴더
                </button>
              )}
            </div>
            <ul>
              {folder.items.map((f: Row) => (
                <li key={f.relative}>
                  <button
                    onClick={() => {
                      if (f.directory) browse(root, f.relative);
                      else {
                        setPath(f.path);
                        open({ path: f.path });
                      }
                    }}
                  >
                    {f.directory ? (
                      <FolderOpen size={16} />
                    ) : (
                      <Scan size={16} />
                    )}
                    {f.name}
                  </button>
                </li>
              ))}
            </ul>
            {folder.truncated && (
              <p>
                최대 500개 항목을 표시합니다. 정확한 하위 경로를 위에 입력할 수
                있습니다.
              </p>
            )}
          </div>
        )}
      </details>
      {error && (
        <div className="alert error" role="alert">
          {error}
        </div>
      )}
      <DatasetEda
        api={api}
        onTarget={(t: string) => {
          setTarget(t);
          setMode("labeled");
          setOffset(0);
        }}
        onRaw={(id: string) => {
          setSource("raw");
          open({ raw: id });
        }}
      />
      <div className="viewer-workspace">
        {session?.local_inventory && (
          <div className="viewer-warning" style={{ gridColumn: "1 / -1" }}>
            {session.local_inventory.split === "test"
              ? "공개 테스트: 추론·입출력 점검용이며 학습에 사용하지 않습니다. "
              : "학습 원본: 파일 목록과 라벨을 확인하세요. "}
            {session.local_inventory.status !== "complete" &&
              `부분 다운로드 또는 완전성 미확인 · 현재 ${session.local_inventory.files}개 / 예상 ${session.local_inventory.expected_files ?? "미확인"}개 파일. 학습 입력으로 사용하지 마세요.`}
          </div>
        )}
        <aside className="viewer-study-panel" aria-label="영상 study 선택">
          <h3>Study 탐색</h3>
          <div
            className="viewer-nav-toggle"
            role="group"
            aria-label="탐색 방식"
          >
            <button
              aria-pressed={navigation === "tree"}
              onClick={() => setNavigation("tree")}
            >
              폴더 구조
            </button>
            <button
              aria-pressed={navigation === "list"}
              onClick={() => setNavigation("list")}
            >
              목록 보기
            </button>
          </div>
          {navigation === "list" && (
            <label>
              데이터 선택
              <select
                aria-label="영상 캐시"
                value={source}
                onChange={(e) => {
                  setSource(e.target.value);
                  setOffset(0);
                }}
              >
                <option value="" disabled>
                  캐시 선택
                </option>
                {catalog.caches.map((c: Row) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
                {catalog.raw?.some((row: Row) => row.files) && (
                  <option value="raw">전체 원본 DICOM · ZIP 포함</option>
                )}
              </select>
            </label>
          )}
          <label className="viewer-search">
            <Search size={16} />
            <input
              aria-label="영상 study 검색"
              placeholder="Study UID 검색"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setOffset(0);
              }}
            />
          </label>
          <select
            aria-label="영상 품질 필터"
            value={mode}
            onChange={(e) => {
              setMode(e.target.value);
              setOffset(0);
            }}
          >
            <option value="all">전체 study</option>
            <option value="missing">
              {source === "raw"
                ? "부분 다운로드·완전성 미확인"
                : "결측 슬라이스 있음"}
            </option>
            <option value="labeled">공식 라벨 있음</option>
          </select>
          <select
            aria-label="양성 소견 필터"
            value={target}
            onChange={(e) => {
              setTarget(e.target.value);
              setOffset(0);
            }}
          >
            <option value="">모든 소견</option>
            {Object.keys(targetNames).map((t) => (
              <option key={t} value={t}>
                {t} 양성
              </option>
            ))}
          </select>
          {navigation === "tree" ? (
            <ViewerTree
              api={api}
              catalog={catalog}
              query={query}
              mode={mode}
              target={target}
              busy={busy}
              session={session}
              onOpen={(body: Row, focus?: Row) => {
                if (body.raw) setSource("raw");
                else if (body.source) setSource(body.source);
                open(body, focus);
              }}
            />
          ) : (
            <>
              <div className="viewer-list-caption">
                <span>{fmt(studies.total)}개</span>
                {listBusy && <LoaderCircle className="spin" size={15} />}
              </div>
              <div className="viewer-study-list" aria-busy={listBusy}>
                {studies.items.map((s: Row) => (
                  <button
                    key={s.id || s.uid}
                    disabled={busy || listBusy}
                    aria-pressed={
                      source === "raw"
                        ? session?.raw_id === s.id
                        : session?.uid === s.uid
                    }
                    onClick={() =>
                      open(
                        source === "raw"
                          ? { raw: s.id }
                          : { source, uid: s.uid },
                      )
                    }
                    title={s.uid}
                  >
                    <span className="viewer-study-number">#{s.index + 1}</span>
                    <strong>…{s.uid.slice(-14)}</strong>
                    <span>
                      {source === "raw" && (
                        <small className="viewer-tag">
                          {s.split === "test" ? "테스트" : "학습"} · {s.files}장
                        </small>
                      )}
                      {s.labeled ? (
                        <small className="viewer-tag">공식 라벨</small>
                      ) : (
                        <small>라벨 미표기</small>
                      )}
                      {s.missing > 0 && (
                        <small className="viewer-tag warning">
                          {source === "raw" ? "미다운로드" : "결측"} {s.missing}
                        </small>
                      )}
                    </span>
                  </button>
                ))}
              </div>
              {!studies.items.length && (
                <p className="viewer-muted">
                  조건에 맞는 캐시 study가 없습니다. 로컬 파일·폴더로 영상을 열
                  수도 있습니다.
                </p>
              )}
              <div className="viewer-pagination">
                <button
                  className="icon-button"
                  aria-label="이전 study 목록"
                  disabled={!offset || listBusy}
                  onClick={() => setOffset(Math.max(0, offset - 40))}
                >
                  <ChevronLeft size={20} />
                </button>
                <span>
                  {studies.total
                    ? `${offset + 1}–${Math.min(offset + 40, studies.total)}`
                    : "0"}
                </span>
                <button
                  className="icon-button"
                  aria-label="다음 study 목록"
                  disabled={offset + 40 >= studies.total || listBusy}
                  onClick={() => setOffset(offset + 40)}
                >
                  <ChevronRight size={20} />
                </button>
              </div>
              {studies.summary && (
                <details className="viewer-dataset-summary">
                  <summary>이 캐시의 품질·라벨 현황</summary>
                  <p>
                    전체 {fmt(studies.summary.studies)} study
                    <br />
                    결측 포함 {fmt(studies.summary.missing_studies)} study
                  </p>
                  <p>미표기 라벨을 음성으로 처리하지 않습니다.</p>
                  <table>
                    <thead>
                      <tr>
                        <th>소견</th>
                        <th>양성</th>
                        <th>음성</th>
                        <th>미표기</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(studies.summary.targets).map(
                        ([t, count]: [string, any]) => (
                          <tr key={t}>
                            <td>{t}</td>
                            <td>{count.positive}</td>
                            <td>{count.negative}</td>
                            <td>{count.unknown}</td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </details>
              )}
            </>
          )}
        </aside>
        <div className="viewer-main">
          {!session ? (
            <div className="viewer-welcome">
              {busy ? (
                <LoaderCircle className="spin" size={34} />
              ) : (
                <Scan size={40} />
              )}
              <h3>
                {busy
                  ? "원본을 준비하고 영상 정보를 읽고 있습니다"
                  : "확인할 study를 선택하세요"}
              </h3>
              <p>
                왼쪽에서 전체 원본 또는 캐시를 선택하세요. ZIP 원본은 첫 열기 때
                해당 study만 해제하므로 잠시 걸릴 수 있습니다.
                <br />
                슬라이스·결측 마스크·공식 라벨을 함께 확인할 수 있습니다.
              </p>
              <div>
                <span>슬라이스 탐색</span>
                <span>대비·확대</span>
                <span>영역 통계</span>
                <span>검토 기록</span>
              </div>
            </div>
          ) : (
            <>
              <header className="viewer-study-heading">
                <div>
                  <span className="eyebrow">
                    {session.kind === "cache"
                      ? "PREPROCESSED CACHE"
                      : session.kind.toUpperCase()}
                  </span>
                  <h3>{session.name}</h3>
                  <code title={session.uid}>
                    {session.uid || "Study UID 없음 · 단독 배열"}
                  </code>
                </div>
                <span className="viewer-tag">
                  {session.series.length} 시리즈
                </span>
              </header>
              <div
                className="viewer-series"
                role="group"
                aria-label="MRI 시리즈"
              >
                {session.series.map((s: Row, i: number) => (
                  <button
                    key={i}
                    aria-pressed={i === slot}
                    onClick={() => chooseSeries(i)}
                  >
                    <strong>{s.name}</strong>
                    <span>
                      {s.slices}장
                      {s.valid
                        ? ` · 유효 ${s.valid.filter(Boolean).length}`
                        : ""}
                    </span>
                  </button>
                ))}
              </div>
              <ImageCanvas
                key={session.id + ":" + slot}
                frame={frame?.slot === slot ? frame : null}
                roi={current ? roi : null}
                onRoi={(box) => {
                  setPlaying(false);
                  setRoi(box);
                }}
                onStep={step}
                busy={!current || busy}
              />
              <div className="viewer-slice-controls">
                <button
                  className="icon-button"
                  aria-label={playing ? "재생 중지" : "슬라이스 재생"}
                  onClick={() => setPlaying((p) => !p)}
                >
                  {playing ? <Pause size={20} /> : <Play size={20} />}
                </button>
                <button
                  className="icon-button"
                  aria-label="이전 슬라이스"
                  disabled={slice === 0}
                  onClick={() => step(-1)}
                >
                  <ChevronLeft size={20} />
                </button>
                <input
                  aria-label="슬라이스 위치"
                  type="range"
                  min="0"
                  max={series.slices - 1}
                  value={slice}
                  onChange={(e) => {
                    setPlaying(false);
                    setSlice(+e.target.value);
                  }}
                />
                <button
                  className="icon-button"
                  aria-label="다음 슬라이스"
                  disabled={slice === series.slices - 1}
                  onClick={() => step(1)}
                >
                  <ChevronRight size={20} />
                </button>
                <span>
                  {slice + 1} / {series.slices}
                </span>
              </div>
              {series.valid && (
                <div
                  className="viewer-mask-strip"
                  aria-label="슬라이스 유효 마스크"
                >
                  {series.valid.map((ok: boolean, i: number) => (
                    <button
                      key={i}
                      className={
                        (ok ? "valid" : "missing") +
                        (slice === i ? " selected" : "")
                      }
                      onClick={() => {
                        setPlaying(false);
                        setSlice(i);
                      }}
                      aria-label={`슬라이스 ${i + 1}: ${ok ? "유효" : "결측"}`}
                      title={`${i + 1}: ${ok ? "유효" : "결측"}`}
                    />
                  ))}
                </div>
              )}
              <div className="viewer-window-controls">
                <label>
                  밝기 최소
                  <input
                    type="number"
                    aria-label="밝기 최소"
                    step="any"
                    value={windowRange?.[0] ?? frame?.window[0] ?? 0}
                    onChange={(e) =>
                      setWindowRange([
                        Number(e.target.value),
                        windowRange?.[1] ?? frame?.window[1] ?? 255,
                      ])
                    }
                  />
                </label>
                <label>
                  밝기 최대
                  <input
                    type="number"
                    aria-label="밝기 최대"
                    step="any"
                    value={windowRange?.[1] ?? frame?.window[1] ?? 255}
                    onChange={(e) =>
                      setWindowRange([
                        windowRange?.[0] ?? frame?.window[0] ?? 0,
                        Number(e.target.value),
                      ])
                    }
                  />
                </label>
                <button
                  className="button secondary"
                  onClick={() => {
                    setWindowRange(null);
                    setInvert(false);
                  }}
                >
                  <RefreshCw size={15} />
                  자동 대비
                </button>
                <label className="viewer-check">
                  <input
                    type="checkbox"
                    checked={invert}
                    onChange={(e) => setInvert(e.target.checked)}
                  />
                  흑백 반전
                </label>
                <span className="viewer-muted">
                  {windowRange
                    ? "범위 고정"
                    : session.kind === "cache"
                      ? "원래 캐시 범위 0–255"
                      : "각 슬라이스 1–99% 자동"}
                </span>
              </div>
              <div className="viewer-preview-actions">
                <button
                  className="button secondary"
                  disabled={contactBusy}
                  onClick={showContact}
                >
                  {contactBusy ? (
                    <LoaderCircle size={16} className="spin" />
                  ) : (
                    <Scan size={16} />
                  )}
                  슬라이스 모아보기
                </button>
                {current && (
                  <a
                    className="button secondary"
                    href={frame.image}
                    download={`kneeops-s${slot + 1}-z${slice + 1}.png`}
                  >
                    <ArrowDownToLine size={16} />
                    현재 PNG
                  </a>
                )}
                <span>정렬: {series.ordering}</span>
              </div>
              {contact && (
                <div className="viewer-contact">
                  <p>{contact.note}</p>
                  <div>
                    {contact.items.map((f: Row) => (
                      <button
                        key={f.slice}
                        onClick={() => {
                          setPlaying(false);
                          setSlice(f.slice);
                        }}
                        aria-label={`미리보기 슬라이스 ${f.slice + 1}`}
                      >
                        <img src={f.image} alt={`슬라이스 ${f.slice + 1}`} />
                        <span>
                          #{f.slice + 1} {!f.valid && "· 결측"}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <SeriesEda
                key={`quality-${session.id}-${slot}`}
                api={api}
                session={session}
                slot={slot}
                onSlice={(z: number) => {
                  setPlaying(false);
                  setSlice(z);
                }}
                onWindow={(v: [number, number]) => {
                  setPlaying(false);
                  setWindowRange(v);
                }}
              />
              <ViewerCompare
                key={`compare-${session.id}`}
                api={api}
                catalog={catalog}
                session={session}
                frame={frame?.slot === slot ? frame : null}
                slot={slot}
                slice={slice}
              />
              <PreprocessingEda
                key={`preprocess-${session.id}-${slot}-${slice}`}
                api={api}
                session={session}
                slot={slot}
                slice={slice}
                frame={current ? frame : null}
              />
              <div className="viewer-analysis-grid">
                <section className="viewer-panel">
                  <div className="viewer-panel-heading">
                    <h3>{roi ? "선택 영역 통계" : "현재 슬라이스 통계"}</h3>
                    {roi && (
                      <button onClick={() => setRoi(null)}>영역 해제</button>
                    )}
                  </div>
                  {frame && !frame.valid ? (
                    <p className="viewer-warning">
                      결측 마스크가 지정된 영상입니다. 값이 있어도 유효 영상으로
                      분석하지 않습니다.
                    </p>
                  ) : (
                    <Histogram
                      info={
                        current ? (roi ? frame?.roi : frame?.stats) : undefined
                      }
                    />
                  )}
                  {frame?.roi && (
                    <p className="viewer-muted">
                      ROI {frame.roi.pixel_box.join(", ")} px{" "}
                      {frame.roi.area_mm2 != null &&
                        `· ${fmt(frame.roi.area_mm2)} mm²`}
                    </p>
                  )}
                  {!!frame?.stats.nonfinite && (
                    <p className="viewer-warning">
                      NaN/Inf {frame.stats.nonfinite}개를 통계에서 제외했습니다.
                    </p>
                  )}
                  <p className="viewer-muted">
                    {series.spacing
                      ? `픽셀 간격 ${series.spacing.join(" × ")} mm`
                      : "물리 간격 없음 · 길이는 픽셀 단위"}
                    <br />
                    통계는 표시 대비·반전 적용 전 픽셀 값입니다.
                  </p>
                </section>
                <section className="viewer-panel">
                  <h3>
                    공식 라벨{" "}
                    <span className="viewer-muted">
                      {session.metadata?.split || "연결 정보 없음"}
                    </span>
                  </h3>
                  {session.metadata ? (
                    <>
                      <div className="viewer-labels">
                        {Object.entries(session.metadata.labels).map(
                          ([t, value]) => (
                            <div
                              className={
                                value === 1
                                  ? "positive"
                                  : value === 0
                                    ? "negative"
                                    : ""
                              }
                              key={t}
                              title={targetNames[t]}
                            >
                              <span>{t}</span>
                              <strong>
                                {value === 1
                                  ? "양성"
                                  : value === 0
                                    ? "음성"
                                    : "미표기"}
                              </strong>
                            </div>
                          ),
                        )}
                      </div>
                      <p className="viewer-muted">
                        출처: {session.metadata.source}
                        <br />
                        미표기는 정답 0이 아닙니다. 라벨이 있는 Gold58은 기존
                        실험에 노출된 진단셋입니다.
                      </p>
                    </>
                  ) : (
                    <p className="viewer-muted">
                      UID와 일치하는 공식 CSV를 찾지 못했습니다. 단독 NPY의
                      라벨을 추정하지 않습니다.
                    </p>
                  )}
                </section>
              </div>
              <details className="viewer-panel viewer-report">
                <summary>
                  학습 판독문 {session.metadata?.report ? "· Report" : "· 없음"}
                </summary>
                <p className="viewer-muted">
                  학습 연구용 텍스트입니다. 공식 테스트에는 Report가 없으며 추론
                  입력으로 사용할 수 없습니다.
                </p>
                <p className="viewer-report-text">
                  {session.metadata?.report ||
                    "해당 study의 학습 판독문이 없습니다."}
                </p>
              </details>
              <section
                className="viewer-panel viewer-review"
                onFocus={() => setPlaying(false)}
              >
                <h3>검토 메모</h3>
                <p className="viewer-muted">
                  현재 시리즈·슬라이스와 ROI·표시 범위·픽셀 해시를 함께
                  저장합니다. 같은 원본 버전의 같은 슬라이스는 마지막 메모로
                  갱신합니다.
                </p>
                <div className="viewer-review-flags">
                  {Object.entries(flags).map(([key, label]) => (
                    <label key={key}>
                      <input
                        type="checkbox"
                        checked={selectedFlags.includes(key)}
                        onChange={(e) =>
                          setSelectedFlags(
                            e.target.checked
                              ? [...selectedFlags, key]
                              : selectedFlags.filter((f) => f !== key),
                          )
                        }
                      />
                      {label}
                    </label>
                  ))}
                </div>
                <textarea
                  disabled={saving}
                  aria-label="영상 검토 메모"
                  maxLength={4000}
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder="예: 슬라이스 중앙부 잘림 여부, 라벨과 판독문의 불일치 등"
                />
                <div className="viewer-review-actions">
                  <button
                    className="button primary"
                    disabled={
                      saving ||
                      !current ||
                      playing ||
                      (!note.trim() && !selectedFlags.length)
                    }
                    onClick={save}
                  >
                    검토 기록 저장
                  </button>
                  <a
                    className="button secondary"
                    href={`/api/viewer/export?session=${session.id}`}
                    download
                  >
                    <ArrowDownToLine size={16} />
                    검토 JSON
                  </a>
                  <span role="status">{saved}</span>
                </div>
                {reviews.length > 0 && (
                  <details>
                    <summary>이 study의 저장 기록 {reviews.length}개</summary>
                    <ul>
                      {reviews.map((r) => (
                        <li key={r.id}>
                          <button
                            onClick={() => {
                              if (r.revision !== session.revision) return;
                              chooseSeries(r.slot);
                              setSlice(r.slice);
                              setRoi(r.frame.roi?.box || null);
                              setWindowRange(r.frame.window);
                              setInvert(r.frame.invert);
                            }}
                            disabled={r.revision !== session.revision}
                          >
                            {r.series} · #{r.slice + 1}
                            {r.revision !== session.revision &&
                              " (이전 파일 버전)"}
                          </button>
                          <span>
                            {r.flags.map((f: string) => flags[f]).join(" · ")}
                          </span>
                          <p>{r.note}</p>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </section>
              <details className="viewer-panel viewer-provenance">
                <summary>원본·전처리·해석 범위</summary>
                <code>{session.source}</code>
                {session.contract && (
                  <p>
                    계약: {session.contract}
                    <br />
                    샘플링: {session.sampling}
                    <br />
                    정규화: {session.normalization}
                  </p>
                )}
                {session.warnings.map((text: string, i: number) => (
                  <p key={i}>{text}</p>
                ))}
                <p>
                  연구·데이터 품질 점검용입니다. 병변 분할이나 모델 예측 결과는
                  포함하지 않습니다.
                </p>
              </details>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
