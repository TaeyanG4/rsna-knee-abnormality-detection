// Unified diff text with added/removed lines marked by both color and the +/- sign.
export default function DiffView({ text }: { text: string }) {
  const lines = text.split("\n");
  return (
    <pre className="source-content diff-view" aria-label="코드 차이">
      {lines.map((line, i) => {
        const kind =
          line.startsWith("+++") || line.startsWith("---")
            ? "file"
            : line.startsWith("@@")
              ? "hunk"
              : line.startsWith("+")
                ? "add"
                : line.startsWith("-")
                  ? "del"
                  : "";
        return (
          <span key={i} className={kind ? "diff-" + kind : undefined}>
            {line + "\n"}
          </span>
        );
      })}
    </pre>
  );
}
