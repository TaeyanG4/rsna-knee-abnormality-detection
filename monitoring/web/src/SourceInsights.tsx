import { useEffect, useState } from "react";
import { ArrowUpRight, Filter, GitCompareArrows, Search } from "lucide-react";

type Api = (path: string, options?: RequestInit) => Promise<any>;
type Row = Record<string, any>;

const KST = (value?: string | null) =>
  value
    ? new Date(value.endsWith("Z") || value.includes("+") ? value : value + "Z").toLocaleDateString("ko-KR", {
        year: "2-digit",
        month: "numeric",
        day: "numeric",
        timeZone: "Asia/Seoul",
      })
    : "";
const INPUT_KINDS: [string, string][] = [
  ["models", "Kaggle 모델"],
  ["datasets", "데이터셋"],
  ["kernels", "노트북 출력"],
  ["competitions", "대회 데이터"],
];
export const inputUrl = (kind: string, ref: string) =>
  "https://www.kaggle.com/" +
  ({ models: "models/", datasets: "datasets/", kernels: "code/", competitions: "competitions/" }[kind] || "") +
  ref.split("/").map(encodeURIComponent).join("/");

export function MetaChips({ meta, kind }: { meta?: Row | null; kind: string }) {
  if (!meta) return null;
  const chips: string[] = [];
  if (meta.votes != null) chips.push(`추천 ${meta.votes}`);
  if (meta.comments != null) chips.push(`댓글 ${meta.comments}`);
  if (meta.last_run) chips.push(`마지막 실행 ${KST(meta.last_run)}`);
  else if (meta.posted_at) chips.push(`게시 ${KST(meta.posted_at)}`);
  return (
    <div className="meta-chips">
      {chips.map((c) => (
        <span key={c}>{c}</span>
      ))}
      {meta.public_score != null && (
        <span className="verified" title="Kaggle 노트북 목록의 best public score. 비공식 웹 API 관측값입니다.">
          공개 점수 {meta.public_score.toFixed(3)}
          {meta.medal ? " · " + meta.medal.toLowerCase() : ""}
        </span>
      )}
      {meta.runtime_s > 0 && <span>실행 {(meta.runtime_s / 60).toFixed(0)}분</span>}
      {meta.public_score == null && meta.score_claim > 0 && (
        <span className="claim" title="노트북 제목에 적힌 점수입니다. 공개 점수가 확인되지 않았습니다.">
          제목 점수 {meta.score_claim.toFixed(3)} · 작성자 주장
        </span>
      )}
      {kind === "notebook" && meta.inputs && (
        <span>입력 {INPUT_KINDS.reduce((n, [k]) => n + (meta.inputs[k]?.length || 0), 0)}개</span>
      )}
    </div>
  );
}

export default function SourceInsights({
  api,
  source,
  onOpen,
  onDiff,
  onFilterInput,
}: {
  api: Api;
  source: Row;
  onOpen: (id: string) => void;
  onDiff: (text: string) => void;
  onFilterInput: (ref: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [options, setOptions] = useState<Row[]>([]);
  const [result, setResult] = useState<Row | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const comparable = ["notebook", "own_notebook", "discussion", "solution"].includes(source.kind);
  const inputs = source.meta?.inputs;
  useEffect(() => {
    setResult(null);
    setQuery("");
    setOptions([]);
    setError("");
  }, [source.id]);
  useEffect(() => {
    if (!comparable || query.trim().length < 2) {
      setOptions([]);
      return;
    }
    const timer = setTimeout(() => {
      const kind = source.kind === "own_notebook" ? "" : source.kind;
      api(`/sources?q=${encodeURIComponent(query)}&kind=${kind}&limit=8&sort=votes`)
        .then((d) => setOptions(d.items.filter((i: Row) => i.id !== source.id)))
        .catch(() => setOptions([]));
    }, 200);
    return () => clearTimeout(timer);
  }, [query, source.id]);
  const compare = async (other: string) => {
    setBusy(true);
    setError("");
    try {
      setResult(await api(`/source/compare?a=${encodeURIComponent(source.id)}&b=${encodeURIComponent(other)}`));
      setOptions([]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="source-insights">
      {inputs && (
        <details className="inputs-panel" open>
          <summary>이 노트북이 쓰는 입력</summary>
          <p className="muted-note">
            Kaggle이 기록한 첨부 입력입니다. 같은 가중치·라벨을 쓰는 노트북을 찾을 때 사용하세요. 비공개·삭제된 입력은 빠집니다.
          </p>
          {INPUT_KINDS.filter(([k]) => inputs[k]?.length).map(([k, label]) => (
            <div className="input-group" key={k}>
              <strong>
                {label} {inputs[k].length}
              </strong>
              <ul>
                {inputs[k].map((ref: string) => (
                  <li key={ref}>
                    <a href={inputUrl(k, ref)} target="_blank" rel="noreferrer">
                      {ref} <ArrowUpRight size={12} />
                    </a>
                    {k !== "competitions" && (
                      <button
                        className="text-button"
                        onClick={() => onFilterInput(ref)}
                        title="자료 라이브러리에서 이 입력을 쓰는 노트북만 봅니다"
                      >
                        <Filter size={12} /> 같은 입력
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          ))}
          {!INPUT_KINDS.some(([k]) => inputs[k]?.length) && <p className="muted-note">기록된 입력이 없습니다.</p>}
        </details>
      )}
      {comparable && (
        <details className="compare-panel-source">
          <summary>
            <GitCompareArrows size={15} /> 다른 자료와 비교
          </summary>
          <p className="muted-note">
            포크 관계를 찾을 때 사용하세요. 현재 저장본끼리 코드·본문 줄 차이와 입력 차이를 보여줍니다. 성능 차이를 뜻하지 않습니다.
          </p>
          <div className="search-field compact">
            <Search size={16} />
            <input
              aria-label="비교할 자료 검색"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="제목·코드로 검색 (2자 이상)"
            />
          </div>
          {options.length > 0 && (
            <ul className="compare-options" role="listbox" aria-label="비교 후보">
              {options.map((o) => (
                <li key={o.id}>
                  <button role="option" aria-selected="false" disabled={busy} onClick={() => compare(o.id)}>
                    <span>{o.title}</span>
                    <small>
                      {o.votes != null ? `추천 ${o.votes}` : ""}
                      {o.score_claim > 0 ? ` · 제목 ${o.score_claim.toFixed(3)}` : ""}
                      {o.body_state !== "full" ? " · 목록만" : ""}
                    </small>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {error && <p className="sync-error">{error}</p>}
          {result && (
            <div className="compare-result" aria-live="polite">
              <p>
                <button className="text-button" onClick={() => onOpen(result.b.id)}>
                  {result.b.title}
                </button>
                와 비교 · 유사도 {(result.similarity * 100).toFixed(1)}% · 줄 <span className="diff-add-text">+{result.added}</span> /{" "}
                <span className="diff-del-text">−{result.removed}</span>
              </p>
              {Object.keys(result.inputs).length > 0 && (
                <table className="mini-table">
                  <thead>
                    <tr>
                      <th>입력</th>
                      <th>이 자료만</th>
                      <th>비교 자료만</th>
                      <th>공통</th>
                    </tr>
                  </thead>
                  <tbody>
                    {INPUT_KINDS.filter(([k]) => result.inputs[k]).map(([k, label]) => (
                      <tr key={k}>
                        <th>{label}</th>
                        {(["only_a", "only_b"] as const).map((side) => (
                          <td key={side}>
                            {result.inputs[k][side].length ? (
                              <ul>
                                {result.inputs[k][side].map((ref: string) => (
                                  <li key={ref}>{ref}</li>
                                ))}
                              </ul>
                            ) : (
                              "—"
                            )}
                          </td>
                        ))}
                        <td>{result.inputs[k].both.length}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <button className="button secondary" onClick={() => onDiff(result.diff || "본문 차이 없음")}>
                <GitCompareArrows size={15} /> 줄 단위 차이 보기{result.truncated ? " (앞부분)" : ""}
              </button>
            </div>
          )}
        </details>
      )}
    </div>
  );
}
