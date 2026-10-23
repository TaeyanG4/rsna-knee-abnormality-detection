type Row = Record<string, any>;
type Api = (path: string, options?: RequestInit) => Promise<any>;

// Transient display buffer, separate from disk training caches. A new session,
// series, window, inversion or ROI invalidates it. Saves still re-read the source.
export class ViewerFrameCache {
  private group = "";
  private bytes = 0;
  private frames = new Map<
    string,
    { value: Row; bytes: number; expires: number }
  >();
  private pending = new Map<
    string,
    { controller: AbortController; promise: Promise<Row> }
  >();

  reset(group = "") {
    for (const p of this.pending.values()) p.controller.abort();
    this.pending.clear();
    this.frames.clear();
    this.bytes = 0;
    this.group = group;
  }

  configure(body: Row) {
    const { slice, ...settings } = body;
    const group = JSON.stringify(settings);
    if (group !== this.group) this.reset(group);
  }

  peek(body: Row): Row | undefined {
    const key = JSON.stringify(body),
      hit = this.frames.get(key);
    if (!hit) return;
    this.frames.delete(key);
    if (hit.expires <= Date.now()) {
      this.bytes -= hit.bytes;
      return;
    }
    this.frames.set(key, hit);
    return hit.value;
  }

  async get(api: Api, body: Row): Promise<Row> {
    const hit = this.peek(body);
    if (hit) return hit;
    const key = JSON.stringify(body),
      old = this.pending.get(key);
    if (old) return old.promise;
    const controller = new AbortController(),
      group = this.group;
    const promise = (async () => {
      const value = await api("/viewer/frame", {
        method: "POST",
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const decoded = new Image();
      decoded.src = value.image;
      await decoded.decode();
      if (controller.signal.aborted || group !== this.group)
        throw new DOMException("Superseded frame", "AbortError");
      value.decodedImage = decoded;
      value.requestKey = key;
      const bytes = decoded.width * decoded.height * 4 + value.image.length * 2;
      this.frames.set(key, { value, bytes, expires: Date.now() + 15000 });
      this.bytes += bytes;
      while (this.frames.size > 24 || this.bytes > 32 * 1024 * 1024) {
        const first = this.frames.keys().next().value!;
        this.bytes -= this.frames.get(first)!.bytes;
        this.frames.delete(first);
      }
      return value;
    })();
    this.pending.set(key, { controller, promise });
    try {
      return await promise;
    } finally {
      if (this.pending.get(key)?.controller === controller)
        this.pending.delete(key);
    }
  }
}
