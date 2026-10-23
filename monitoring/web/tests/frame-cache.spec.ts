import { test, expect } from "@playwright/test";
import { ViewerFrameCache } from "../src/viewer-frame-cache";

test("frame buffer deduplicates requests, expires entries and bounds decoded memory", async () => {
  const original = globalThis.Image;
  const clock = Date.now;
  let time = clock(),
    calls = 0;
  try {
    globalThis.Image = class {
      width = 1024;
      height = 1024;
      src = "";
      async decode() {}
    } as any;
    Date.now = () => time;
    const cache = new ViewerFrameCache();
    const body = { session: "fixture", slot: 0, slice: 0, invert: false };
    cache.configure(body);
    const api = async () => {
      calls++;
      return { image: "fixture" };
    };
    const [a, b] = await Promise.all([
      cache.get(api, body),
      cache.get(api, body),
    ]);
    expect(a).toBe(b);
    expect(calls).toBe(1);
    for (let z = 1; z <= 9; z++) await cache.get(api, { ...body, slice: z });
    expect(cache.peek(body)).toBeUndefined(); // 32 MiB budget, including decoded pixels.
    const recent = { ...body, slice: 9 };
    expect(cache.peek(recent)).toBeDefined();
    time += 15001;
    expect(cache.peek(recent)).toBeUndefined();
    await cache.get(api, recent);
    cache.configure({ ...body, invert: true });
    expect(cache.peek(recent)).toBeUndefined();
    cache.reset();
  } finally {
    globalThis.Image = original;
    Date.now = clock;
  }
});

test("a superseded request cannot repopulate another study buffer", async () => {
  const original = globalThis.Image;
  try {
    globalThis.Image = class {
      width = 16;
      height = 16;
      src = "";
      async decode() {}
    } as any;
    const cache = new ViewerFrameCache();
    const body = { session: "old", slot: 0, slice: 0 };
    cache.configure(body);
    let finish!: (value: any) => void;
    let signal!: AbortSignal;
    const api = async (_: string, options?: RequestInit) => {
      signal = options!.signal!;
      return new Promise((resolve) => {
        finish = resolve;
      });
    };
    const request = cache.get(api, body);
    const rejected = expect(request).rejects.toMatchObject({
      name: "AbortError",
    });
    cache.configure({ ...body, session: "new" });
    expect(signal.aborted).toBe(true);
    finish({ image: "fixture" });
    await rejected;
    expect(cache.peek(body)).toBeUndefined();
  } finally {
    globalThis.Image = original;
  }
});
