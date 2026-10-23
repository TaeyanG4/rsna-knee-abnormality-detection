import React, { useEffect, useState } from "react";
import { BarChart3, Download, RefreshCw, Microscope } from "lucide-react";
type Row = Record<string, any>;
export const n = (v: any, d = 2) =>
  v == null
    ? "—"
    : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: d });
export function exportJson(data: Row, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function Bars({ items }: { items: Row[] }) {
  const max = Math.max(1, ...items.map((x) => x.count));
  return (
    <div className="eda-bars">
      {items.map((x, i) => (
        <div key={i}>
          <span>{x.name}</span>
          <div>
            <i style={{ width: `${(x.count / max) * 100}%` }} />
          </div>
          <strong>{n(x.count, 0)}</strong>
        </div>
      ))}
    </div>
  );
}
function Distribution({ label, data }: { label: string; data: Row }) {
  return (
    <section>
      <h4>{label}</h4>
      <p>
        n={n(data.n, 0)} · 최소 {n(data.min)} · 중앙 {n(data.median)} · P95{" "}
        {n(data.p95)} · 최대 {n(data.max)}
      </p>
      <Bars items={data.bins} />
    </section>
  );
}

export function DatasetEda({ api, onTarget, onRaw }: Row) {
  const [expanded, setExpanded] = useState(false),
    [data, setData] = useState<Row | null>(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [split, setSplit] = useState("train"),
    [generation, setGeneration] = useState(0);
  useEffect(() => {
    if (!expanded) return;
    let live = true;
    setBusy(true);
    setError("");
    api("/viewer/eda/dataset")
      .then((d: Row) => live && setData(d))
      .catch((e: Error) => live && setError(e.message))
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, [expanded, generation]);
  const selected = data?.splits[split];
  return (
    <details
      className="viewer-panel dataset-eda"
      onToggle={(e) => {
        if (e.target === e.currentTarget) setExpanded(e.currentTarget.open);
      }}
    >
      <summary>
        <BarChart3 size={20} />
        전체 데이터 EDA · 라벨·촬영 구성·원본/캐시 범위
      </summary>
      {expanded && (
        <>
          <p>
            공식 CSV와 ZIP 목록, 캐시 마스크를 집계합니다. 전체 영상 픽셀을 읽지
            않습니다.
          </p>
          <div className="eda-actions">
            <button
              className="button secondary"
              disabled={busy}
              onClick={() => setGeneration(generation + 1)}
            >
              <RefreshCw size={15} />
              현황 다시 계산
            </button>
            {data && (
              <button
                className="button secondary"
                onClick={() => exportJson(data, "kneeops-dataset-eda.json")}
              >
                <Download size={15} />
                EDA JSON
              </button>
            )}
          </div>
        </>
      )}
      {busy && <p role="status">라벨·촬영 메타데이터를 집계하고 있습니다…</p>}
      {error && <p role="alert">{error}</p>}
      {data && (
        <>
          <div className="eda-kpis">
            <div>
              <strong>{n(data.splits.train.studies, 0)}</strong>
              <span>학습 study</span>
            </div>
            <div>
              <strong>{n(data.splits.test.studies, 0)}</strong>
              <span>공개 테스트</span>
            </div>
            <div>
              <strong>{n(data.labeled_studies, 0)}</strong>
              <span>공식 라벨 있는 study</span>
            </div>
            <div>
              <strong>{n(data.report_present, 0)}</strong>
              <span>학습 Report 있음</span>
            </div>
          </div>
          <p className="viewer-warning">
            {data.warnings[0]} {data.warnings[1]}
          </p>
          <div className="eda-grid">
            <section>
              <h3>소견별 라벨 분포</h3>
              <p>소견을 누르면 영상 탐색에 해당 양성 필터를 적용합니다.</p>
              <div className="eda-table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>소견</th>
                      <th>양성</th>
                      <th>음성</th>
                      <th>미표기</th>
                      <th>라벨 내 양성률</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.targets.map((t: string) => {
                      const c = data.labels[t],
                        known = c.positive + c.negative;
                      return (
                        <tr key={t}>
                          <th>
                            <button onClick={() => onTarget(t)}>{t}</button>
                          </th>
                          <td>{n(c.positive, 0)}</td>
                          <td>{n(c.negative, 0)}</td>
                          <td>{n(c.unknown, 0)}</td>
                          <td>
                            {known
                              ? `${n((c.positive / known) * 100, 1)}%`
                              : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
            <section>
              <h3>원본 구성</h3>
              <label>
                집계 데이터
                <select
                  aria-label="EDA 집계 데이터"
                  value={split}
                  onChange={(e) => setSplit(e.target.value)}
                >
                  <option value="train">학습</option>
                  <option value="test">공개 테스트</option>
                </select>
              </label>
              <p>
                시리즈 {n(selected.series, 0)} · DICOM{" "}
                {n(selected.dicom_files, 0)} · 준비{" "}
                {n(selected.prepared_studies, 0)}/{n(selected.raw_studies, 0)}{" "}
                study
              </p>
              <h4>촬영 방향 · 시리즈 수 기준</h4>
              <Bars items={selected.planes} />
              <details>
                <summary>방향 × 수분 민감도 × 지방억제</summary>
                <p>
                  Fluid=수분 민감도 · FS=지방억제. 1: 있음, 0: 없음, ?: 미표기.
                </p>
                <Bars items={selected.protocols} />
              </details>
            </section>
          </div>
          <div className="eda-grid">
            <Distribution
              label="Study당 시리즈 수"
              data={selected.series_per_study}
            />
            <Distribution
              label="Study당 DICOM 수 · 인덱스 범위"
              data={selected.files_per_study}
            />
          </div>
          <details>
            <summary>소견 동시 양성 행렬</summary>
            <p>
              셀: 동시 양성 수 / 두 라벨 모두 알려진 study 수. 미표기를 음성으로
              채우지 않습니다.
            </p>
            <div className="eda-table-wrap">
              <table className="eda-cooccurrence">
                <thead>
                  <tr>
                    <th>소견</th>
                    {data.targets.map((t: string) => (
                      <th key={t}>{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.targets.map((t: string, i: number) => (
                    <tr key={t}>
                      <th>{t}</th>
                      {data.copositive[i].map((c: Row, j: number) => (
                        <td
                          key={j}
                          style={{
                            backgroundColor: `rgba(23,129,112,${c.known ? (c.both_positive / c.known) * 0.5 : 0})`,
                          }}
                          title={`${t} + ${data.targets[j]}: 동시 양성 ${c.both_positive}, 관측 ${c.known}`}
                        >
                          {c.both_positive}/{c.known}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
          <details>
            <summary>촬영 누락·메타데이터 정합성</summary>
            <p>
              train/test Study UID 교집합 {data.train_test_uid_overlap} · CSV
              중복 study 행 {selected.duplicate_study_rows} · study CSV에 없는
              시리즈 study {selected.orphan_series_studies} · 원본 목록에 없는
              study {selected.missing_raw}
            </p>
            <p>
              {data.warnings[2]} {data.warnings[3]}
            </p>
            {Object.entries(selected.missing_planes).map(
              ([plane, value]: [string, any]) => (
                <details key={plane}>
                  <summary>
                    {plane}가 CSV에 없는 study {n(value.count, 0)}건
                  </summary>
                  <div className="eda-example-list">
                    {value.examples.map((x: Row) => (
                      <button
                        key={x.uid}
                        disabled={!x.raw}
                        onClick={() => onRaw(x.raw)}
                        title={x.uid}
                      >
                        …{x.uid.slice(-18)} 영상 확인
                      </button>
                    ))}
                  </div>
                  <p>
                    최대30건 표시. 촬영 프로토콜 차이일 수 있으므로 영상으로
                    확인하세요.
                  </p>
                </details>
              ),
            )}
          </details>
          <details>
            <summary>캐시별 study 포함 범위·슬롯 결측</summary>
            {data.caches.map((c: Row) => (
              <section key={c.id}>
                <h4>{c.name}</h4>
                <p>
                  {c.shape.join(" × ")} · 학습 UID 누락 {c.missing_train} · 학습
                  CSV 외 UID {c.extra_studies} · 결측 포함 {c.missing_studies}{" "}
                  study
                </p>
                <div className="eda-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Slot</th>
                        <th>전체 결측 study</th>
                        <th>결측 슬라이스 / 전체</th>
                      </tr>
                    </thead>
                    <tbody>
                      {c.slots.map((s: Row) => (
                        <tr key={s.name}>
                          <td>{s.name}</td>
                          <td>{n(s.empty_studies, 0)}</td>
                          <td>
                            {n(s.missing_slices, 0)} / {n(s.total_slices, 0)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            ))}
          </details>
          <details>
            <summary>출처·집계 시점·해석 범위</summary>
            <p>
              {new Date(data.observed_at).toLocaleString("ko-KR")} ·{" "}
              {data.scope}
            </p>
            {data.sources.map((s: Row) => (
              <p key={s.path}>
                <code>{s.path}</code> · {s.rows}행<br />
                <code>SHA256 {s.sha256}</code>
              </p>
            ))}
            <Distribution
              label="학습 Report 문자 수"
              data={data.report_characters}
            />
          </details>
        </>
      )}
    </details>
  );
}

export function SeriesEda({ api, session, slot, onSlice, onWindow }: Row) {
  const [data, setData] = useState<Row | null>(null),
    [samples, setSamples] = useState(16),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [onlyFlags, setOnlyFlags] = useState(false);
  // Parent keys this component by session + slot, so no stale audit can replace a new series.
  const run = async () => {
    setBusy(true);
    setError("");
    try {
      setData(
        await api("/viewer/eda/series", {
          method: "POST",
          body: JSON.stringify({ session: session.id, slot, samples }),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const s = session.series[slot],
    h = s.header || {},
    pos = s.positions_mm || [];
  return (
    <details className="viewer-panel series-eda">
      <summary>
        <Microscope size={19} />
        시리즈 품질·촬영 정보
      </summary>
      <div className="eda-metadata">
        <span>
          해상도{" "}
          <b>
            {s.rows} × {s.columns}
          </b>
        </span>
        <span>
          슬라이스 <b>{s.slices}</b>
        </span>
        <span>
          픽셀 간격{" "}
          <b>
            {s.spacing?.map((x: number) => n(x, 3)).join(" × ") || "없음"}{" "}
            {s.spacing ? "mm" : ""}
          </b>
        </span>
        <span>
          두께 <b>{n(h.SliceThickness)} mm</b>
        </span>
        <span>
          TR / TE{" "}
          <b>
            {n(h.RepetitionTime)} / {n(h.EchoTime)} ms
          </b>
        </span>
        <span>
          자기장 / Flip angle{" "}
          <b>
            {n(h.MagneticFieldStrength)} T / {n(h.FlipAngle)}°
          </b>
        </span>
      </div>
      <p>
        {s.header_scope
          ? "촬영 파라미터는 해당 시리즈 첫 DICOM 기준입니다."
          : "캐시/NPY에는 원본 촬영 파라미터가 없을 수 있습니다."}{" "}
        물리 위치 확인 {pos.length}장.
      </p>
      <div className="eda-actions">
        <label>
          표본 슬라이스
          <select
            aria-label="품질 검사 표본 수"
            value={samples}
            onChange={(e) => setSamples(+e.target.value)}
          >
            {[8, 16, 32, 64].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <button className="button primary" disabled={busy} onClick={run}>
          {busy ? "픽셀 검사 중…" : "시리즈 품질 검사"}
        </button>
        {data && (
          <>
            <button
              className="button secondary"
              onClick={() => exportJson(data, "kneeops-series-quality.json")}
            >
              검사 JSON
            </button>
            <button
              className="button secondary"
              disabled={!data.recommended_window}
              onClick={() => onWindow(data.recommended_window)}
            >
              표본 공통 대비 적용
            </button>
          </>
        )}
      </div>
      {error && <p role="alert">{error}</p>}
      {data && (
        <>
          <p>
            {data.sampled_slices}/{data.total_slices}장 검사 ·{" "}
            {data.all_slices ? "이 시리즈 전체" : "균등 표본"} · 점검 후보{" "}
            {data.flagged_samples}장
          </p>
          <p>
            촬영 범위 {n(data.geometry.coverage_mm)} mm · 위치 간격 중앙{" "}
            {n(data.geometry.step_mm.median, 3)} mm · 위치 중복{" "}
            {data.geometry.duplicate_positions} · 간격 불균일{" "}
            {data.geometry.irregular_steps} · FOV{" "}
            {data.geometry.fov_mm?.map((x: number) => n(x)).join(" × ") ||
              "미확인"}{" "}
            mm
          </p>
          <div className="eda-slice-chart" aria-label="슬라이스 평균 강도 추이">
            {data.frames.map((f: Row) => {
              const max = Math.max(
                1,
                ...data.frames.map((r: Row) => Math.abs(r.mean || 0)),
              );
              return (
                <button
                  key={f.slice}
                  title={`#${f.slice + 1} 평균 ${n(f.mean)} ${f.flags.join(", ")}`}
                  onClick={() => onSlice(f.slice)}
                  className={f.flags.length ? "flagged" : ""}
                >
                  <i
                    style={{
                      height: `${Math.max(3, (Math.abs(f.mean || 0) / max) * 80)}px`,
                    }}
                  />
                  <small>{f.slice + 1}</small>
                </button>
              );
            })}
          </div>
          <label className="eda-check">
            <input
              type="checkbox"
              checked={onlyFlags}
              onChange={(e) => setOnlyFlags(e.target.checked)}
            />
            점검 후보만 표시
          </label>
          <div className="eda-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>슬라이스</th>
                  <th>평균 / 표준편차</th>
                  <th>0 값 %</th>
                  <th>경계 에너지</th>
                  <th>인접 변화</th>
                  <th>점검 사유</th>
                </tr>
              </thead>
              <tbody>
                {data.frames
                  .filter((f: Row) => !onlyFlags || f.flags.length)
                  .map((f: Row) => (
                    <tr key={f.slice}>
                      <td>
                        <button onClick={() => onSlice(f.slice)}>
                          #{f.slice + 1} 열기
                        </button>
                      </td>
                      <td>
                        {n(f.mean)} / {n(f.std)}
                      </td>
                      <td>{n(f.zero_percent)}</td>
                      <td>{n(f.edge_energy, 4)}</td>
                      <td>{n(f.adjacent_change, 4)}</td>
                      <td>{f.flags.join(" · ") || "—"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="viewer-muted">{data.notes.join(" ")}</p>
        </>
      )}
    </details>
  );
}

export function PreprocessingEda({ api, session, slot, slice, frame }: Row) {
  const [resolution, setResolution] = useState(288),
    [crop, setCrop] = useState(100),
    [normal, setNormal] = useState("p1"),
    [data, setData] = useState<Row | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const run = async () => {
    setBusy(true);
    setError("");
    try {
      setData(
        await api("/viewer/eda/preview", {
          method: "POST",
          body: JSON.stringify({
            session: session.id,
            slot,
            slice,
            resolution,
            crop_percent: crop,
            normalization: normal,
          }),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <details className="viewer-panel preprocessing-eda">
      <summary>전처리 미리보기 · 해상도·crop·정규화</summary>
      <p>
        한 장의 변환을 비교합니다. 원본 파일과 학습 설정을 변경하지 않습니다.
      </p>
      <div className="eda-actions">
        <label>
          출력 해상도
          <select
            aria-label="전처리 출력 해상도"
            disabled={busy}
            value={resolution}
            onChange={(e) => {
              setResolution(+e.target.value);
              setData(null);
            }}
          >
            {[128, 224, 256, 288, 336, 384, 512].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          중앙 crop (%)
          <select
            aria-label="중앙 crop 비율"
            disabled={busy}
            value={crop}
            onChange={(e) => {
              setCrop(+e.target.value);
              setData(null);
            }}
          >
            {[100, 85, 70, 50, 25].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label>
          강도 정규화
          <select
            aria-label="전처리 정규화"
            disabled={busy}
            value={normal}
            onChange={(e) => {
              setNormal(e.target.value);
              setData(null);
            }}
          >
            <option value="p1">1–99 백분위</option>
            <option value="p2">2–98 백분위</option>
            <option value="minmax">최소–최대</option>
          </select>
        </label>
        <button
          className="button primary"
          disabled={busy || !frame?.valid}
          onClick={run}
        >
          {busy ? "변환 중…" : "전처리 비교 실행"}
        </button>
      </div>
      {error && <p role="alert">{error}</p>}
      {data && (
        <>
          <div className="eda-image-pair">
            <figure>
              {frame && <img src={frame.image} alt="변환 전 현재 슬라이스" />}
              <figcaption>
                현재 표시 · {data.original_shape.join(" × ")}
              </figcaption>
            </figure>
            <figure>
              <img src={data.image} alt="전처리 미리보기" />
              <figcaption>
                전처리 결과 · crop {data.crop_percent}% · {data.normalization} ·{" "}
                {data.output_shape.join(" × ")}
              </figcaption>
            </figure>
          </div>
          <p>
            보존 면적 {n(data.retained_area_percent, 1)}% · crop FOV{" "}
            {data.crop_fov_mm?.map((x: number) => n(x)).join(" × ") ||
              "물리 간격 없음"}{" "}
            {data.crop_fov_mm ? "mm" : ""} · 정규화 범위{" "}
            {data.window.map((x: number) => n(x)).join("–")}
          </p>
          <p>
            1장 입력 크기: uint8 {n(data.uint8_bytes / 1024, 1)} KiB · float32{" "}
            {n(data.float32_bytes / 1024, 1)} KiB. 모델·배치·활성화를 포함한 GPU
            메모리 추정이 아닙니다.
          </p>
          {data.upsampled && (
            <p className="viewer-warning">
              원래 crop보다 확대했습니다. 새 영상 정보가 생기지 않습니다.
            </p>
          )}
          <p>{data.note}</p>
          <div className="eda-actions">
            <a
              className="button secondary"
              href={data.image}
              download="kneeops-preprocessing-preview.png"
            >
              미리보기 PNG
            </a>
            <button
              className="button secondary"
              onClick={() => {
                const { image, ...record } = data;
                exportJson(record, "kneeops-preprocessing-preview.json");
              }}
            >
              변환 JSON
            </button>
          </div>
        </>
      )}
    </details>
  );
}
