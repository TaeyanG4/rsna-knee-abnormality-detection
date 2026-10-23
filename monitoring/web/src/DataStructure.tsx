import React, { useEffect, useState } from "react";
type Row = Record<string, any>;
type Api = (path: string, options?: RequestInit) => Promise<any>;
export const bytes = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${(n / 1e6).toFixed(1)} MB`;

export function DatasetDescription({ data }: { data: Row }) {
  const train = data.studies.filter((s: Row) => s.split === "train");
  const test = data.studies.filter((s: Row) => s.split === "test");
  return (
    <details className="data-details raw-description" open>
      <summary>전체 데이터 구조·설명</summary>
      <p><a href="https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data" target="_blank" rel="noreferrer">Kaggle 공식 데이터 설명 ↗</a></p>
      <p>
        하나의 study는 한 검사이며, 그 안에 촬영 방향·설정이 다른 MRI 시리즈와
        여러 장의 DICOM 슬라이스가 있습니다.
      </p>
      <div className="raw-layout">
        <pre aria-label="원본 폴더 구조">{`대회 원본 ZIP\n├─ train_series/  학습 ${train.length.toLocaleString()} study\n│  └─ StudyInstanceUID/\n│     └─ SeriesInstanceUID/\n│        └─ *.dcm  원본 MRI 슬라이스\n├─ test_series/   공개 테스트 ${test.length.toLocaleString()} study\n│  └─ StudyInstanceUID/SeriesInstanceUID/*.dcm\n└─ 공식 CSV 5개`}</pre>
        <dl className="raw-file-descriptions">
          <dt>train.csv</dt>
          <dd>
            학습 study, 판독문(Report), 제공된 공식 라벨. 미표기는 음성과
            다릅니다.
          </dd>
          <dt>train_series.csv</dt>
          <dd>
            학습 study·시리즈 UID, 촬영 방향(Anatomical_Plane), 수분
            민감도(Fluid_Sensitive), 지방억제(Fat_Suppression)를 연결합니다.
          </dd>
          <dt>test.csv · test_series.csv</dt>
          <dd>
            공개 테스트 study·시리즈 목록. 학습 정답과 추론용 Report는 없습니다.
          </dd>
          <dt>sample_submission.csv</dt>
          <dd>제출할 UID·예측 열의 형식 예시입니다.</dd>
        </dl>
      </div>
      <p className="data-hint">
        UID는 폴더를 연결하는 식별자입니다. 파일명 순서는 해부학적 슬라이스
        순서가 아니며 뷰어는 DICOM 위치 정보를 읽어 정렬합니다. 비공개 채점
        데이터는 이 ZIP에 포함되지 않습니다.
      </p>
    </details>
  );
}

export function StudyStructure({ api, id }: { api: Api; id: string }) {
  const [expanded, setExpanded] = useState(false),
    [data, setData] = useState<Row | null>(null);
  const [series, setSeries] = useState(""),
    [offset, setOffset] = useState(0);
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!expanded) return;
    let live = true;
    setBusy(true);
    setError("");
    api(
      `/data/structure?raw=${id}&series=${encodeURIComponent(series)}&offset=${offset}`,
    )
      .then((value) => live && setData(value))
      .catch((e) => live && setError(e.message))
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, [expanded, series, offset, id]);
  return (
    <details
      className="raw-series-tree"
      onToggle={(e) => setExpanded(e.currentTarget.open)}
    >
      <summary>시리즈·파일 구조 및 촬영 설명</summary>
      {error && <p role="alert">{error}</p>}
      {busy && <p role="status">파일 목록 읽는 중…</p>}
      {data && (
        <>
          <p className="data-hint">
            공식 CSV의 촬영 방향·수분 민감도·지방억제 정보입니다. 진단 결과가
            아니며, DICOM의 상세 촬영명은 영상 열기 후 표시합니다. 목록 탐색은
            압축을 해제하지 않습니다.
          </p>
          <div className="raw-series-buttons">
            {data.series.map((s: Row) => (
              <button
                className={
                  "button " + (series === s.series ? "primary" : "secondary")
                }
                key={s.series}
                onClick={() => {
                  setSeries(s.series);
                  setOffset(0);
                }}
              >
                <strong>{s.description || "촬영명 미제공"}</strong>
                <small>
                  {s.files}장 · {bytes(s.bytes)}
                </small>
                <code>{s.series}</code>
              </button>
            ))}
          </div>
          {series && (
            <div
              className="raw-member-list"
              aria-label="시리즈 파일 목록"
              aria-busy={busy}
            >
              {data.files.map((f: Row) => (
                <div key={f.path}>
                  <code>{f.path.split("/").pop()}</code>
                  <span>{bytes(f.bytes)}</span>
                </div>
              ))}
              <div className="raw-pagination">
                <button
                  className="button secondary"
                  disabled={busy || offset === 0}
                  onClick={() => setOffset(offset - 30)}
                >
                  이전 파일
                </button>
                <span>
                  {data.total ? offset + 1 : 0}–
                  {Math.min(offset + 30, data.total)} / {data.total}개
                </span>
                <button
                  className="button secondary"
                  disabled={busy || offset + 30 >= data.total}
                  onClick={() => setOffset(offset + 30)}
                >
                  다음 파일
                </button>
              </div>
            </div>
          )}
          <p className="data-hint">
            sag: 시상면 · cor: 관상면 · tra/ax: 횡단면 · PD: 양성자밀도 · T1/T2:
            강조영상 · FS: 지방억제. 실제 촬영명 표기는 시리즈마다 다를 수
            있습니다.
          </p>
        </>
      )}
    </details>
  );
}
