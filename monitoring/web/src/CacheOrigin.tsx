import { ExternalLink } from "lucide-react";
import registry from "../../kneeops/cache_origins.json";

type Origin = {
  kind: string;
  creator_account: string;
  notebook_url: string | null;
  raw_data_url: string;
  summary: string;
  reference?: { label: string; url: string; role: string };
};

export default function CacheOrigin({ manifestSha }: { manifestSha?: string }) {
  // Bundled evidence also works while an older server is busy downloading;
  // this UI update never requires interrupting that server.
  const origin = (registry.by_manifest_sha256 as Record<string, Origin>)[
    manifestSha || ""
  ];
  if (!origin)
    return (
      <p className="data-cache-origin-unknown">
        출처 미확인 · 공식 캐시 여부와 생성 계정이 등록되지 않았습니다.
      </p>
    );
  return (
    <div className="data-cache-origin">
      <div className="data-inline">
        <span className="badge">자체 생성 · 비공식 캐시</span>
        <span>
          생성 계정 <strong>{origin.creator_account}</strong>
        </span>
      </div>
      <p>{origin.summary}</p>
      <div className="data-source-links">
        {origin.notebook_url ? (
          <a href={origin.notebook_url} target="_blank" rel="noreferrer">
            생성 노트북 바로가기 <ExternalLink size={14} />
          </a>
        ) : (
          <span>이 PC에서 training/src/raptor_corpus.py로 생성</span>
        )}
        <a href={origin.raw_data_url} target="_blank" rel="noreferrer">
          공식 원본 데이터 <ExternalLink size={14} />
        </a>
        {origin.reference && (
          <a
            href={origin.reference.url}
            title={origin.reference.role}
            target="_blank"
            rel="noreferrer"
          >
            {origin.reference.label} <ExternalLink size={14} />
          </a>
        )}
      </div>
      <p className="data-hint">
        비공개 노트북으로 기록되어 있습니다. 해당 계정 또는 공유 권한으로
        로그인해야 열 수 있습니다.
      </p>
      {origin.reference && (
        <p className="data-hint">
          OrthoDiffusion은 참고 연구 프로젝트이며, 이 캐시는 위 생성 계정의
          전처리 결과입니다.
        </p>
      )}
    </div>
  );
}
