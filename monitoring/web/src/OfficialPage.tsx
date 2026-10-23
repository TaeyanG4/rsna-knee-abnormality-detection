import React, { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { ArrowUpRight, BookOpen, Languages, LoaderCircle } from "lucide-react";
import "katex/dist/katex.min.css";
import "./official.css";

type Row = Record<string, any>;
type Api = (path: string, options?: RequestInit) => Promise<any>;
type Language = "en" | "ko" | "both";
const COMPETITION = "rsna-knee-abnormality-detection";
export const officialTitles: Record<string, string> = {
  Description: "대회 개요",
  rules: "대회 규칙",
  Evaluation: "평가 방식",
  Timeline: "대회 일정",
  "data-description": "데이터 설명",
  "Code Requirements": "코드·제출 요건",
  "Efficiency Prize Evaluation": "효율성 상 평가",
  Prizes: "상금·수상 요건",
  abstract: "한눈에 보기",
  Acknowledgements: "기여자",
};
const pageName = (source: Row) => source.title.split(" · ").at(-1) as string;
const title = (source: Row) => officialTitles[pageName(source)] || source.title;
const date = (value: string) =>
  new Date(value).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" });
const isToday = (value: string) =>
  new Date(value).toLocaleDateString("en-CA", { timeZone: "Asia/Seoul" }) ===
  new Date().toLocaleDateString("en-CA", { timeZone: "Asia/Seoul" });

function formatMarkdown(value: string) {
  // Normalize display only; never alter the saved source or its checksum.
  return value
    .replace(
      /<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi,
      (_, level, text) => `\n${"#".repeat(Number(level))} ${text}\n`,
    )
    .replace(/<br\s*\/?\s*>/gi, "\n\n")
    .replace(/<\/?(?:h[1-7]|div)[^>]*>/gi, "")
    .replace(/^(#{1,6})(?=[^#\s])/gm, "$1 ")
    .replace(/\\{1,2}\[/g, () => "\n$$\n")
    .replace(/\\{1,2}\]/g, () => "\n$$\n")
    .replace(
      /\\{1,2}\((.*?)\\{1,2}\)/g,
      (_, text) =>
        "`" +
        text.replace(/\\text\{([^}]+)\}/g, "$1").replace(/\\max/g, "max ") +
        "`",
    );
}

export function OfficialDocument({ source }: { source: Row }) {
  const [language, setLanguage] = useState<Language>(() => {
    try {
      const value = localStorage.getItem("kneeops-official-language");
      return value === "en" || value === "both" ? value : "ko";
    } catch {
      return "ko";
    }
  });
  const available = source.translation?.status === "available";
  const shown = available ? language : "en";
  const select = (value: Language) => {
    setLanguage(value);
    try {
      localStorage.setItem("kneeops-official-language", value);
    } catch {
      /* Storage can be disabled. */
    }
  };
  const link = (url: string) => {
    try {
      const parsed = new URL(
        url,
        `https://www.kaggle.com/competitions/${source.competition}/`,
      );
      return ["https:", "http:"].includes(parsed.protocol) ? parsed.href : "";
    } catch {
      return "";
    }
  };
  const render = (text: string, lang: "en" | "ko") => (
    <section
      className="official-language-column"
      lang={lang}
      key={lang}
      aria-label={lang === "ko" ? "한국어 번역 본문" : "영어 원문 본문"}
    >
      {shown === "both" && (
        <h3 className="official-column-label">
          {lang === "ko" ? "한국어 번역" : "English · 원문"}
        </h3>
      )}
      <div className="official-prose">
        <Markdown
          remarkPlugins={[
            remarkGfm,
            [remarkMath, { singleDollarTextMath: false }],
          ]}
          rehypePlugins={[rehypeKatex]}
          skipHtml
          urlTransform={link}
          components={{
            img: () => null,
            a: ({ href, children }) =>
              href ? (
                <a href={href} target="_blank" rel="noreferrer">
                  {children}
                </a>
              ) : (
                <span>{children}</span>
              ),
          }}
        >
          {formatMarkdown(text)}
        </Markdown>
      </div>
    </section>
  );
  return (
    <div className="official-document">
      <div className="official-language-bar">
        <span>
          <Languages size={18} /> 문서 언어
        </span>
        <div
          className="segmented-control"
          role="group"
          aria-label="공식 문서 언어"
        >
          {(
            [
              ["en", "English"],
              ["ko", "한국어"],
              ["both", "나란히 보기"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              aria-pressed={shown === value}
              disabled={value !== "en" && !available}
              onClick={() => select(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <p className="official-translation-note" role="status">
        {available
          ? source.translation.disclaimer +
            " 원문과 같은 저장본을 기준으로 작성했습니다."
          : source.translation?.message || "영어 원문입니다."}
      </p>
      {available && source.translation.notes?.length > 0 && (
        <details className="official-notes">
          <summary>번역·원문 표기 참고</summary>
          {source.translation.notes.map((note: string) => (
            <p key={note}>{note}</p>
          ))}
        </details>
      )}
      <div
        className={
          "official-columns " + (shown === "both" ? "side-by-side" : "")
        }
      >
        {shown !== "ko" && render(source.revision.content, "en")}
        {shown !== "en" &&
          available &&
          render(source.translation.content, "ko")}
      </div>
    </div>
  );
}

export default function OfficialPage({
  api,
  competitions,
}: {
  api: Api;
  competitions: Row[];
}) {
  const [competition, setCompetition] = useState(COMPETITION);
  const [items, setItems] = useState<Row[]>([]);
  const [source, setSource] = useState<Row | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const listSequence = useRef(0),
    sourceSequence = useRef(0);
  const open = async (id: string, revision?: string) => {
    const n = ++sourceSequence.current;
    setLoading(true);
    setError("");
    setSelectedId(id);
    try {
      const result = await api(
        "/source?id=" +
          encodeURIComponent(id) +
          (revision ? "&revision=" + encodeURIComponent(revision) : ""),
      );
      if (sourceSequence.current === n) setSource(result);
    } catch (e) {
      if (sourceSequence.current === n) {
        setError((e as Error).message);
        setSource(null);
      }
    } finally {
      if (sourceSequence.current === n) setLoading(false);
    }
  };
  useEffect(() => {
    const n = ++listSequence.current;
    ++sourceSequence.current;
    setLoading(true);
    setError("");
    setSource(null);
    setItems([]);
    setSelectedId("");
    api(
      "/sources?kind=official&limit=200&competition=" +
        encodeURIComponent(competition),
    )
      .then((result) => {
        if (n !== listSequence.current) return;
        const keys = Object.keys(officialTitles);
        const sorted = result.items.sort(
          (a: Row, b: Row) =>
            (keys.indexOf(pageName(a)) < 0 ? 99 : keys.indexOf(pageName(a))) -
            (keys.indexOf(pageName(b)) < 0 ? 99 : keys.indexOf(pageName(b))),
        );
        setItems(sorted);
        if (sorted.length) open(sorted[0].id);
        else setLoading(false);
      })
      .catch((e) => {
        if (n === listSequence.current) {
          setError(e.message);
          setLoading(false);
        }
      });
    return () => {
      ++listSequence.current;
      ++sourceSequence.current;
    };
  }, [competition]);

  return (
    <div className="official-page">
      <div className="official-page-tools">
        <div>
          <BookOpen size={20} />
          <p>개요와 규칙을 원문·번역으로 확인하세요.</p>
        </div>
        <label>
          대회
          <select
            aria-label="공식 안내 대회"
            value={competition}
            onChange={(e) => setCompetition(e.target.value)}
          >
            <option value={COMPETITION}>RSNA Knee · 현재</option>
            {competitions
              .filter((c) => c.slug !== COMPETITION)
              .map((c) => (
                <option value={c.slug} key={c.slug}>
                  {c.title}
                </option>
              ))}
          </select>
        </label>
      </div>
      {error && (
        <div className="alert error" role="alert">
          {error}
        </div>
      )}
      <label className="official-mobile-picker">
        문서
        <select
          aria-label="공식 문서 선택"
          value={selectedId}
          onChange={(e) => open(e.target.value)}
        >
          {items.map((item) => (
            <option value={item.id} key={item.id}>
              {title(item)}
            </option>
          ))}
        </select>
      </label>
      <div className="official-reader">
        <nav className="official-page-list" aria-label="공식 문서 목록">
          {items.map((item) => (
            <button
              key={item.id}
              aria-current={selectedId === item.id ? "page" : undefined}
              className={selectedId === item.id ? "selected" : ""}
              onClick={() => open(item.id)}
            >
              <span>{title(item)}</span>
              <small>{pageName(item)}</small>
            </button>
          ))}
        </nav>
        <article className="official-paper" aria-busy={loading}>
          {loading ? (
            <div className="loading">
              <LoaderCircle className="spin" size={21} />
              문서를 불러오고 있습니다…
            </div>
          ) : source ? (
            <>
              <header className="official-paper-heading">
                <div>
                  <span className="eyebrow">OFFICIAL COMPETITION GUIDE</span>
                  <h2>{title(source)}</h2>
                  <p>원문 수집 {date(source.revision.observed_at)} KST</p>
                  <p className="official-added">
                    추가 {date(source.added_at)} KST{" "}
                    {isToday(source.added_at) && (
                      <span className="new-badge">NEW</span>
                    )}
                  </p>
                </div>
                <a
                  className="button secondary"
                  href={`https://www.kaggle.com/competitions/${source.competition}/${pageName(source) === "rules" ? "rules" : pageName(source) === "data-description" ? "data" : "overview"}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  Kaggle 원문 <ArrowUpRight size={15} />
                </a>
              </header>
              {source.revisions.length > 1 && (
                <label className="official-revision">
                  원문 저장본
                  <select
                    aria-label="공식 문서 저장본"
                    value={source.revision.id}
                    onChange={(e) => open(source.id, e.target.value)}
                  >
                    {source.revisions.map((r: Row) => (
                      <option key={r.id} value={r.id}>
                        {date(r.observed_at)} · {r.sha.slice(0, 10)}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <OfficialDocument
                key={source.id + source.revision.id}
                source={source}
              />
              <details className="official-source-details">
                <summary>출처·번역 기록</summary>
                <p>
                  영어 원문 SHA256: <code>{source.revision.sha}</code>
                </p>
                <p>처음 추가: {date(source.added_at)} KST</p>
                {source.translation?.status === "available" && (
                  <p>
                    {source.translation.translator} ·{" "}
                    {source.translation.translated_at} · 본문 전체 번역
                  </p>
                )}
              </details>
            </>
          ) : (
            <div className="empty">
              <h3>저장된 공식 문서가 없습니다</h3>
              <p>
                현재 대회는 Kaggle 새로고침, 과거 대회는 과거 대회 메뉴의 자료
                수집을 실행하세요.
              </p>
            </div>
          )}
        </article>
      </div>
    </div>
  );
}
