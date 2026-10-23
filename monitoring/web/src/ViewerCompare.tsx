import React, { useEffect, useState } from "react";
import { n } from "./ViewerEda";
type Row = Record<string, any>;
function plane(series: Row): string {
  const o = series.orientation;
  if (o?.length === 6 && o.every((x: any) => typeof x === "number")) {
    const normal = [
      o[1] * o[5] - o[2] * o[4],
      o[2] * o[3] - o[0] * o[5],
      o[0] * o[4] - o[1] * o[3],
    ].map(Math.abs);
    if (Math.max(...normal) > 0.8)
      return ["SAG", "COR", "AX"][normal.indexOf(Math.max(...normal))];
  }
  return (
    series.name
      .toUpperCase()
      .match(/(?:^|[ _-])(SAG|COR|AX)(?:[ _-]|$)/)?.[1] || ""
  );
}
export default function ViewerCompare({
  api,
  catalog,
  session,
  frame,
  slot,
  slice,
}: Row) {
  const raw = catalog.raw?.find((r: Row) => r.uid === session.uid && r.files);
  const [expanded, setExpanded] = useState(false),
    [source, setSource] = useState(
      raw && session.kind === "cache" ? "raw" : catalog.caches[0]?.id || "",
    ),
    [other, setOther] = useState<Row | null>(null),
    [image, setImage] = useState<Row | null>(null),
    [rslot, setRslot] = useState(0),
    [rslice, setRslice] = useState(0),
    [sync, setSync] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    if (!expanded || !source || !session.uid || (source === "raw" && !raw))
      return;
    let live = true;
    setBusy(true);
    setError("");
    setOther(null);
    setImage(null);
    api("/viewer/open", {
      method: "POST",
      body: JSON.stringify(
        source === "raw" ? { raw: raw.id } : { source, uid: session.uid },
      ),
    })
      .then((d: Row) => {
        if (!live) return;
        if (d.uid !== session.uid)
          throw new Error("비교 영상의 Study UID가 다릅니다.");
        setOther(d);
        const direction = plane(session.series[slot]);
        const initial = Math.max(
          0,
          direction
            ? d.series.findIndex((s: Row) => plane(s) === direction)
            : 0,
        );
        setRslot(initial);
        setRslice(Math.floor(d.series[initial].slices / 2));
      })
      .catch((e: Error) => live && setError(e.message))
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, [expanded, source, session.id]);
  useEffect(() => {
    if (sync && other)
      setRslice(
        Math.round(
          (slice / Math.max(1, session.series[slot].slices - 1)) *
            (other.series[rslot].slices - 1),
        ),
      );
  }, [sync, other, rslot, slice, slot]);
  useEffect(() => {
    if (!expanded || !other) return;
    let live = true;
    setImage((previous) => (previous?.slot === rslot ? previous : null));
    setError("");
    const timer = setTimeout(
      () =>
        api("/viewer/frame", {
          method: "POST",
          body: JSON.stringify({
            session: other.id,
            slot: rslot,
            slice: rslice,
          }),
        })
          .then(async (d: Row) => {
            const decoded = new Image();
            decoded.src = d.image;
            await decoded.decode();
            if (live) setImage(d);
          })
          .catch((e: Error) => live && setError(e.message)),
      16,
    );
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [expanded, other, rslot, rslice]);
  return (
    <details
      className="viewer-panel viewer-compare"
      onToggle={(e) => {
        if (e.target === e.currentTarget) setExpanded(e.currentTarget.open);
      }}
    >
      <summary>같은 study 비교 · 원본 ↔ 캐시</summary>
      <p className="viewer-warning">
        Study UID만 일치시킵니다. 시리즈·촬영 방향·해부 위치는 직접 확인하세요.
        자동 정합·픽셀 차감은 하지 않습니다.
      </p>
      <div className="eda-actions">
        <label>
          비교 데이터
          <select
            aria-label="비교 데이터"
            value={source}
            onChange={(e) => setSource(e.target.value)}
          >
            {raw && <option value="raw">전체 원본 DICOM</option>}
            {catalog.caches.map((c: Row) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        {other && (
          <label>
            비교 시리즈
            <select
              aria-label="비교 시리즈"
              value={rslot}
              onChange={(e) => {
                const i = +e.target.value;
                setRslot(i);
                setRslice(Math.floor(other.series[i].slices / 2));
              }}
            >
              {other.series.map((s: Row, i: number) => (
                <option key={i} value={i}>
                  {s.name} · {s.slices}장
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {busy && <p role="status">같은 study의 비교 원본을 준비하고 있습니다…</p>}
      {error && <p role="alert">{error}</p>}
      {other && (
        <>
          <label className="eda-check">
            <input
              type="checkbox"
              checked={sync}
              onChange={(e) => setSync(e.target.checked)}
            />
            상대 슬라이스 위치 맞춤 · 해부학적 동기화 아님
          </label>
          <label>
            비교 슬라이스 {rslice + 1}/{other.series[rslot].slices}
            <input
              aria-label="비교 슬라이스 위치"
              type="range"
              min={0}
              max={other.series[rslot].slices - 1}
              value={rslice}
              disabled={sync}
              onChange={(e) => setRslice(+e.target.value)}
            />
          </label>
          <div className="eda-image-pair">
            <figure>
              {frame ? (
                <img src={frame.image} alt="현재 영상 비교 A" />
              ) : (
                <p>현재 영상 읽는 중…</p>
              )}
              <figcaption>
                A · {session.kind} · {session.series[slot].name} · #
                {(frame?.slice ?? slice) + 1}
                <br />
                {session.series[slot].rows} × {session.series[slot].columns}
              </figcaption>
            </figure>
            <figure>
              {image ? (
                <img src={image.image} alt="대조 영상 비교 B" />
              ) : (
                <p>비교 영상 읽는 중…</p>
              )}
              <figcaption>
                B · {other.kind} · {other.series[rslot].name} · #
                {(image?.slice ?? rslice) + 1}
                <br />
                {other.series[rslot].rows} × {other.series[rslot].columns}
              </figcaption>
            </figure>
          </div>
          <p>
            A는 현재 대비, B는 자동 대비를 사용합니다. 원본 MRI와 정규화된 uint8
            캐시의 강도 값은 직접 비교할 수 없습니다.
          </p>
          {image && !image.valid && (
            <p className="viewer-warning">B는 결측 마스크 슬라이스입니다.</p>
          )}
          <p>
            원본 픽셀 간격 A{" "}
            {session.series[slot].spacing
              ?.map((v: number) => n(v, 3))
              .join(" × ") || "미확인"}{" "}
            / B{" "}
            {other.series[rslot].spacing
              ?.map((v: number) => n(v, 3))
              .join(" × ") || "미확인"}{" "}
            mm
          </p>
        </>
      )}
    </details>
  );
}
