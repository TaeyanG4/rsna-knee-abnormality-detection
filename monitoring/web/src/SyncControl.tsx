import { useEffect, useState } from "react";
import { CalendarClock, LoaderCircle } from "lucide-react";
import "./sync-control.css";

type Api = (path: string, options?: RequestInit) => Promise<any>;
type Schedule = {
  enabled: boolean;
  minutes: number;
  due: boolean;
  next_due: string | null;
  last_sync_attempt: string | null;
  running: { kind: string; progress: string } | null;
  note: string;
  coverage?: any;
};

const INTERVALS = [15, 30, 60, 120, 180, 360, 720, 1440];
const span = (m: number) =>
  m < 60 ? `${m}분` : m % 60 ? `${(m / 60).toFixed(2)}시간` : `${m / 60}시간`;
const kst = (value?: string | null) =>
  value
    ? new Date(value).toLocaleString("ko-KR", {
        month: "numeric",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "Asia/Seoul",
      })
    : "기록 없음";
const n = (value?: number) => (value ?? 0).toLocaleString("ko-KR");

export default function SyncControl({
  api,
  refreshKey,
  onChange,
  showCoverage = true,
}: {
  api: Api;
  refreshKey: string;
  onChange?: (enabled: boolean, minutes: number) => void;
  showCoverage?: boolean;
}) {
  const [status, setStatus] = useState<Schedule | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const load = () =>
    api("/schedule")
      .then((s: Schedule) => {
        setStatus(s);
        setError("");
      })
      .catch((e: Error) => setError(e.message));
  useEffect(() => {
    load();
    const timer = setInterval(load, 30000);
    return () => clearInterval(timer);
  }, [refreshKey]);
  const save = async (enabled: boolean, minutes: number) => {
    const previous = status;
    // Show the choice at once; a controlled input would otherwise flick back until the reply.
    setStatus((s) => (s ? { ...s, enabled, minutes } : s));
    setSaving(true);
    setError("");
    try {
      const next: Schedule = await api("/schedule", {
        method: "PUT",
        body: JSON.stringify({ enabled, minutes }),
      });
      setStatus(next);
      onChange?.(next.enabled, next.minutes);
    } catch (e) {
      setStatus(previous);
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };
  if (!status)
    return (
      <section className="sync-control" aria-label="정기 수집" aria-busy="true">
        <span className="sync-status">
          {error || "정기 수집 상태를 확인하는 중…"}
        </span>
      </section>
    );
  const intervals = INTERVALS.includes(status.minutes)
    ? INTERVALS
    : [...INTERVALS, status.minutes].sort((a, b) => a - b);
  const nb = status.coverage?.notebooks;
  const topics = status.coverage?.discussions;
  const lb = status.coverage?.leaderboard;
  const state = status.running
    ? `수집 중 · ${status.running.progress}`
    : !status.enabled
      ? "꺼짐 · 수동 새로고침만 사용"
      : status.due
        ? "곧 수집을 시작합니다"
        : `다음 수집 ${kst(status.next_due)}`;
  return (
    <section className="sync-control" aria-label="정기 수집">
      <div className="sync-main">
        <label className="sync-switch">
          <input
            type="checkbox"
            role="switch"
            checked={status.enabled}
            disabled={saving}
            aria-describedby="sync-state"
            onChange={(e) => save(e.target.checked, status.minutes)}
          />
          <span className="sync-track" aria-hidden="true">
            <span className="sync-thumb" />
          </span>
          <CalendarClock size={17} />
          <strong>정기 수집</strong>
        </label>
        <select
          aria-label="수집 간격"
          value={status.minutes}
          disabled={saving}
          onChange={(e) => save(status.enabled, Number(e.target.value))}
        >
          {intervals.map((m) => (
            <option key={m} value={m}>
              {span(m)}마다
            </option>
          ))}
        </select>
        <span id="sync-state" className="sync-status" aria-live="polite">
          {status.running && <LoaderCircle className="spin" size={14} />}
          {state}
          <span className="sync-last">
            마지막 {kst(status.last_sync_attempt)} KST
          </span>
        </span>
      </div>
      {showCoverage && nb && topics && (
        <dl className="sync-coverage">
          <div>
            <dt>공개 노트북</dt>
            <dd>
              Kaggle {n(nb.live)}개 · 관련 {n(nb.relevant)}개 · 소스{" "}
              {n(nb.sources)}개
              {nb.pending_new + nb.pending_changed > 0 && (
                <small>
                  {" "}
                  · 대기 {n(nb.pending_new + nb.pending_changed)}
                </small>
              )}
            </dd>
          </div>
          <div>
            <dt>토론</dt>
            <dd>
              {n(topics.live || topics.registered)}개 · 원글·댓글{" "}
              {n(topics.bodies)}개
              {topics.pending > 0 && <small> · 대기 {n(topics.pending)}</small>}
            </dd>
          </div>
          {lb?.own && (
            <div>
              <dt>공개 LB</dt>
              <dd>
                {n(lb.own.rank)}위 / {n(lb.teams)}팀
              </dd>
            </div>
          )}
        </dl>
      )}
      <small className="sync-note">
        {status.note} 한 번에 받는 본문 수에 상한이 있어 대기분은 다음 수집에서
        이어 받습니다.
      </small>
      {error && (
        <small className="sync-error" role="alert">
          {error}
        </small>
      )}
    </section>
  );
}
