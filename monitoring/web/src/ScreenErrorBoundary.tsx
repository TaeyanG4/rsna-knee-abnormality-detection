import React, { Component, type ReactNode } from "react";
import { RefreshCw } from "lucide-react";

// Suspense covers pending imports, not rejected imports or render errors.
// Keep the surrounding navigation mounted when one screen fails to load.
export default class ScreenErrorBoundary extends Component<
  { children: ReactNode; name?: string },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error) {
    console.error("KneeOps screen could not load", error);
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <section className="screen-load-error" role="alert">
        <h2>{this.props.name || "화면"}을 불러오지 못했습니다</h2>
        <p>
          앱이 업데이트되었거나 화면을 읽는 중 오류가 발생했습니다. 새로고침하면
          현재 버전으로 다시 엽니다.
        </p>
        <button className="button primary" onClick={() => location.reload()}>
          <RefreshCw size={18} /> 화면 새로고침
        </button>
        <p className="screen-load-hint">
          저장된 데이터와 다운로드는 유지됩니다. 저장하지 않은 검토 메모는 다시
          입력해야 할 수 있습니다.
        </p>
      </section>
    );
  }
}
