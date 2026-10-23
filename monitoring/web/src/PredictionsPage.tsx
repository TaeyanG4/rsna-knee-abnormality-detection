import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, BarChart3, LoaderCircle, RefreshCw, Search } from "lucide-react";
import "./analysis.css";

type Api = (path: string, options?: RequestInit) => Promise<any>;
type Row = Record<string, any>;

const OFFICIAL = "storage/competition/train.csv";
const fmt = (v?: number | null, d = 3) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(d));
// Sequential single hue: five bins, darker = higher. Text stays in text tokens.
const aucBin = (v?: number | null) => (v == null ? "" : v >= 0.9 ? "q5" : v >= 0.8 ? "q4" : v >= 0.7 ? "q3" : v >= 0.6 ? "q2" : "q1");
const corBin = (v?: number | null) => (v == null ? "" : v >= 0.9 ? "q5" : v >= 0.75 ? "q4" : v >= 0.6 ? "q3" : v >= 0.4 ? "q2" : "q1");
const short = (id: string) => id.split("/").slice(-2).join("/");

export default function PredictionsPage({ api }: { api: Api }) {
  const [files, setFiles] = useState<Row | null>(null);
  const [chosen, setChosen] = useState<string[]>([]);
  const [label, setLabel] = useState(OFFICIAL);
  const [filter, setFilter] = useState("");
  const [result, setResult] = useState<Row | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const load = (refresh = false) =>
    api("/predictions/files" + (refresh ? "?refresh=true" : ""))
      .then(setFiles)
      .catch((e: Error) => setError(e.message));
  useEffect(() => {
    load();
  }, []);
  const visible = useMemo(
    () =>
      (files?.predictions || []).filter((f: Row) =>
        (f.folder + "/" + f.name).toLowerCase().includes(filter.toLowerCase()),
      ),
    [files, filter],
  );
  const toggle = (id: string) =>
    setChosen((c) => (c.includes(id) ? c.filter((x) => x !== id) : c.length >= 4 ? c : [...c, id]));
  const run = async () => {
    setBusy(true);
    setError("");
    try {
      setResult(await api("/predictions/compare", { method: "POST", body: JSON.stringify({ files: chosen, label }) }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const names = result ? [...result.files.map((f: Row) => f.id), ...(result.label?.macro?.blend != null ? ["blend"] : [])] : [];
  const nameOf = (id: string) => (id === "blend" ? "동일 가중 순위 평균" : short(id));
  return (
    <div className="predictions">
      <div className="alert">
        <AlertTriangle size={18} />
        <span>
          로컬 예측 CSV(StudyInstanceUID + 소견 열)를 비교하는 진단 도구입니다. 결과는 고른 라벨 파일 기준이며 대회 점수가 아닙니다. blend는
          가중치를 맞추지 않은 동일 가중 순위 평균입니다.
        </span>
      </div>
      <div className="prediction-layout">
        <section className="panel prediction-picker" aria-label="비교할 파일 선택">
          <div className="panel-title">
            <h2>예측 파일</h2>
            <button className="icon-button" aria-label="파일 목록 새로고침" onClick={() => load(true)}>
              <RefreshCw size={16} />
            </button>
          </div>
          <div className="search-field compact">
            <Search size={16} />
            <input aria-label="예측 파일 검색" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="폴더·파일 이름" />
          </div>
          <p className="muted-note">
            {files ? `${files.predictions.length}개 발견 · 최대 4개 선택 (${chosen.length}/4)` : "파일을 찾는 중…"}
          </p>
          <ul className="file-list">
            {visible.map((f: Row) => (
              <li key={f.id}>
                <label className={chosen.includes(f.id) ? "selected" : ""}>
                  <input
                    type="checkbox"
                    checked={chosen.includes(f.id)}
                    disabled={!chosen.includes(f.id) && chosen.length >= 4}
                    onChange={() => toggle(f.id)}
                  />
                  <span>
                    <strong>{f.name}</strong>
                    <small>
                      {f.folder} · {f.targets.length} target · {(f.bytes / 1024).toFixed(0)} KB
                    </small>
                  </span>
                </label>
              </li>
            ))}
          </ul>
          <label className="field">
            평가 라벨
            <select value={label} onChange={(e) => setLabel(e.target.value)}>
              <option value="">라벨 없이 모델 간 상관만</option>
              {(files?.labels || []).map((f: Row) => (
                <option key={f.id} value={f.id}>
                  {f.id === OFFICIAL ? "공식 라벨 (Gold58, 노출된 진단셋)" : short(f.id)}
                </option>
              ))}
            </select>
          </label>
          <button className="button primary" disabled={!chosen.length || busy} onClick={run}>
            {busy ? <LoaderCircle className="spin" size={16} /> : <BarChart3 size={16} />}
            비교 실행
          </button>
          {error && (
            <p className="sync-error" role="alert">
              {error}
            </p>
          )}
        </section>
        <section className="prediction-results" aria-live="polite">
          {!result ? (
            <div className="empty">
              <BarChart3 size={28} />
              <h3>예측 파일을 고르고 비교를 실행하세요</h3>
              <p>공통 study만 사용합니다. 같은 라벨로 학습한 모델은 점수가 부풀려집니다.</p>
            </div>
          ) : (
            <>
              {result.warnings.map((w: string) => (
                <div className="alert warn" key={w}>
                  <AlertTriangle size={16} />
                  <span>{w}</span>
                </div>
              ))}
              <p className="muted-note">
                공통 study {result.common_studies}개
                {result.label && ` · 라벨 ${result.label.name} (${result.label.studies} study, 평균 target ${result.label.computable_targets.length}개)`}
              </p>
              {result.label && (
                <div className="stat-tiles">
                  {names.map((n) => {
                    const ci = result.label.bootstrap?.[n];
                    return (
                      <div className={"stat-tile" + (n === "blend" ? " blend" : "")} key={n}>
                        <span className="stat-label" title={n}>
                          {nameOf(n)}
                        </span>
                        <strong>{fmt(result.label.macro[n])}</strong>
                        <small>{ci ? `95% 구간 ${fmt(ci.low)}–${fmt(ci.high)}` : "구간 계산 불가"}</small>
                      </div>
                    );
                  })}
                </div>
              )}
              {result.label?.bootstrap?.blend_vs_best && (
                <p className="delta-note">
                  blend − 단독 최고({nameOf(result.label.bootstrap.blend_vs_best.reference)}) 차이의 study 단위 bootstrap 95% 구간:{" "}
                  <strong>
                    {fmt(result.label.bootstrap.blend_vs_best.low, 4)} ~ {fmt(result.label.bootstrap.blend_vs_best.high, 4)}
                  </strong>{" "}
                  · 양수 비율 {(result.label.bootstrap.blend_vs_best.fraction_positive * 100).toFixed(0)}% ({result.label.bootstrap.blend_vs_best.reps}회)
                </p>
              )}
              {result.label && (
                <div className="panel table-wrap">
                  <table className="heat-table" aria-label="소견별 AUC">
                    <caption>소견별 AUC · 색이 진할수록 높음 (0.6/0.7/0.8/0.9 구간)</caption>
                    <thead>
                      <tr>
                        <th scope="col">소견</th>
                        <th scope="col">양성/음성</th>
                        {names.map((n) => (
                          <th scope="col" key={n} title={n}>
                            {nameOf(n)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.label.per_target.map((r: Row) => {
                        const best = Math.max(...names.filter((n) => n !== "blend").map((n) => r.auc[n] ?? -1));
                        return (
                          <tr key={r.target}>
                            <th scope="row">
                              {r.target}
                              {r.label_mode !== "binary" && <small> ({r.label_mode})</small>}
                            </th>
                            <td className="count">
                              {r.n_pos}/{r.n_neg}
                            </td>
                            {names.map((n) => (
                              <td
                                key={n}
                                className={"heat " + aucBin(r.auc[n]) + (n !== "blend" && r.auc[n] === best && best >= 0 ? " best" : "")}
                                title={`${r.target} · ${nameOf(n)} · AUC ${fmt(r.auc[n], 4)} · 양성 ${r.n_pos} / 음성 ${r.n_neg}`}
                              >
                                {fmt(r.auc[n])}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
              {result.correlations.length > 0 && (
                <div className="panel table-wrap">
                  <table className="heat-table" aria-label="모델 간 순위 상관">
                    <caption>모델 간 Spearman 순위 상관 · 낮을수록 서로 다른 오류를 낼 가능성 (공통 study 기준)</caption>
                    <thead>
                      <tr>
                        <th scope="col">비교 쌍</th>
                        <th scope="col">평균</th>
                        {result.targets.map((t: string) => (
                          <th scope="col" key={t} className="rotate">
                            <span>{t}</span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.correlations.map((c: Row) => (
                        <tr key={c.a + c.b}>
                          <th scope="row" className="pair" title={`${c.a} ↔ ${c.b}`}>
                            {short(c.a)}
                            <br />↔ {short(c.b)}
                          </th>
                          <td className={"heat " + corBin(c.macro)}>{fmt(c.macro, 2)}</td>
                          {result.targets.map((t: string) => (
                            <td key={t} className={"heat " + corBin(c.per_target[t])} title={`${t} · ρ ${fmt(c.per_target[t], 3)}`}>
                              {fmt(c.per_target[t], 2)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <div className="panel table-wrap">
                <table className="mini-table" aria-label="파일 요약">
                  <thead>
                    <tr>
                      <th>파일</th>
                      <th>study</th>
                      <th>target</th>
                      <th>값 범위</th>
                      <th>빈 값</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.files.map((f: Row) => (
                      <tr key={f.id}>
                        <td title={f.id}>{f.id}</td>
                        <td>{f.rows}</td>
                        <td>{f.targets.length}</td>
                        <td>
                          {fmt(f.min, 2)} ~ {fmt(f.max, 2)}
                          {f.outside_probability && " (확률 범위 밖)"}
                        </td>
                        <td>{f.missing}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
