import { test, expect } from "@playwright/test";

// The schedule API is mocked so this test never changes the person's real settings.
test("library schedule toggle applies immediately and matches settings", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  let state: Record<string, any> = {
    enabled: false,
    minutes: 60,
    due: false,
    next_due: null,
    last_sync_attempt: "2026-10-02T01:34:32+00:00",
    running: null,
    history: [],
    note: "프로그램이 실행 중일 때만 수집합니다.",
    coverage: {
      notebooks: { live: 963, relevant: 218, sources: 120, versioned: 100, registered: 250, pending_new: 3, pending_changed: 1 },
      discussions: { live: 143, registered: 143, bodies: 140, pending: 3 },
      leaderboard: { teams: 4809, own: { rank: 1356, score: 0.941 } },
    },
  };
  const puts: Record<string, any>[] = [];
  await page.route("**/api/schedule", async (route) => {
    if (route.request().method() === "PUT") {
      const body = route.request().postDataJSON();
      puts.push(body);
      state = {
        ...state,
        enabled: body.enabled,
        minutes: body.minutes,
        next_due: body.enabled ? "2026-10-02T03:34:32+00:00" : null,
      };
    }
    await route.fulfill({ json: state });
  });
  await page.goto("/#research");
  const control = page.getByRole("region", { name: "정기 수집" });
  await expect(control.getByText("꺼짐 · 수동 새로고침만 사용")).toBeVisible();
  await expect(control.getByText(/Kaggle 963개/)).toBeVisible();
  await expect(control.getByText("1,356위 / 4,809팀")).toBeVisible();
  const toggle = control.getByRole("switch", { name: /정기 수집/ });
  await expect(toggle).not.toBeChecked();
  await control.locator(".sync-switch").click();
  await expect(toggle).toBeChecked();
  await expect(control.locator("#sync-state")).toContainText("다음 수집");
  expect(puts[0]).toEqual({ enabled: true, minutes: 60 });
  await control.getByLabel("수집 간격").selectOption("120");
  await expect.poll(() => puts.length).toBe(2);
  expect(puts[1]).toEqual({ enabled: true, minutes: 120 });
  await expect(control.locator(".sync-track")).toHaveCSS("background-color", "rgb(22, 133, 115)");
  await page.screenshot({ path: "../../.local/ui-sync-library.png", fullPage: false });
  await page.goto("/#settings");
  const settings = page.getByRole("region", { name: "정기 수집" });
  await expect(settings.getByRole("switch", { name: /정기 수집/ })).toBeChecked();
  await expect(settings.getByLabel("수집 간격")).toHaveValue("120");
  await expect(page.getByText("수집 간격 (시간)")).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#research");
  await page.reload();
  await expect(page.getByRole("button", { name: "메뉴 열기" })).toHaveAttribute("aria-expanded", "false");
  await page.waitForTimeout(400);
  await expect(page.getByRole("region", { name: "정기 수집" })).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
  await page.screenshot({ path: "../../.local/ui-sync-mobile.png", fullPage: false });
  expect(errors).toEqual([]);
});
