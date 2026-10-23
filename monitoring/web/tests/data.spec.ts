import { test, expect } from "@playwright/test";

test("data page, cache inventory, saved destination and download preview only", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  let starts = 0;
  await page.route("**/api/data/downloads/*/start", (route) => {
    starts++;
    return route.fulfill({ json: { status: "queued" } });
  });
  await page.goto("/#data");
  await expect(
    page.getByRole("heading", { level: 1, name: "학습 데이터" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "기존 캐시로 학습 준비" }),
  ).toBeVisible();
  const path = page.getByLabel("다운로드 폴더", { exact: true });
  const original = await path.inputValue();
  await path.fill(original + "-preview-only");
  await expect(
    page.getByRole("button", { name: "전체 다운로드 준비", exact: true }),
  ).toBeDisabled();
  await path.fill(original);
  await page
    .getByRole("button", { name: "전체 다운로드 준비", exact: true })
    .click();
  const preview = page.getByRole("region", { name: "다운로드 미리보기" });
  await expect(preview).toBeVisible();
  await expect(preview).toContainText(original);
  await expect(preview).toContainText("자동 압축 해제 없음");
  expect(starts).toBe(0);
  await page.screenshot({
    path: "../../.local/ui-data-preview.png",
    fullPage: false,
  });
  await page.getByRole("button", { name: "미리보기 닫기" }).click();
  await page.locator(".data-study-picker summary").click();
  await expect(page.getByLabel("StudyInstanceUID 검색")).toBeVisible();
  const checks = page.locator(".data-study-list input[type=checkbox]");
  await expect(checks.first()).toBeVisible();
  for (let i = 0; i < 5; i++) await checks.nth(i).check();
  await expect(checks.nth(5)).toBeDisabled();
  await expect(page.locator(".data-selected button")).toHaveCount(5);
  expect(errors).toEqual([]);
});

test("data preview start/cancel and settings flow against isolated fixtures", async ({
  page,
}) => {
  let destination = "D:\\KneeData";
  let status = "planned";
  let started = 0;
  const plan = () => ({
    id: "fixture",
    mode: "metadata",
    destination,
    studies: [],
    file_count: 5,
    estimated_bytes: 9000000,
    required_free_bytes: 1100000000,
    free_bytes: 5000000000,
    space_ok: true,
    conflict_count: 0,
    estimate_note: "선택한 CSV만 받습니다.",
  });
  const setting = () => ({
    destination,
    default_destination: "D:\\DefaultData",
    cache_manifest: "",
    free_bytes: 5000000000,
  });
  await page.route("**/api/data**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path === "/api/data")
      return route.fulfill({
        json: {
          settings: setting(),
          official: { bytes: 569764475800 },
          caches: [],
        },
      });
    if (path === "/api/data/studies")
      return route.fulfill({ json: { items: [], total: 0 } });
    if (path === "/api/data/settings") {
      destination = route.request().postDataJSON().destination;
      return route.fulfill({ json: setting() });
    }
    if (path === "/api/data/preview") return route.fulfill({ json: plan() });
    // The page also checks for a local full archive; this fixture has none.
    if (path === "/api/data/library")
      return route.fulfill({ json: { archives: [], studies: [], total: 0 } });
    if (path === "/api/data/downloads")
      return route.fulfill({
        json: [
          {
            id: "fixture",
            status,
            plan: plan(),
            file_count: 5,
            progress: { completed_files: 0, current_bytes: 200 },
            created_at: "2026-10-01T00:00:00Z",
          },
        ],
      });
    if (path.endsWith("/start")) {
      started++;
      status = "running";
      return route.fulfill({ json: { status } });
    }
    if (path.endsWith("/cancel")) {
      status = "cancelled";
      return route.fulfill({ json: { status } });
    }
    throw Error("Unexpected data request " + path);
  });
  await page.goto("/#data");
  await page.getByRole("button", { name: "기본값", exact: true }).click();
  await page.getByRole("button", { name: "경로 저장", exact: true }).click();
  await expect(page.getByLabel("다운로드 폴더", { exact: true })).toHaveValue(
    "D:\\DefaultData",
  );
  await page
    .getByRole("button", { name: "CSV 다운로드 준비", exact: true })
    .click();
  await page
    .getByRole("button", { name: "다운로드 시작", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "중지", exact: true }),
  ).toBeVisible();
  expect(started).toBe(1);
  await page.getByRole("button", { name: "중지", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "재시도 / 이어받기" }),
  ).toBeVisible();
});

test("data mobile readability and overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#data");
  await expect(page.getByLabel("다운로드 폴더", { exact: true })).toBeVisible();
  await page.screenshot({
    path: "../../.local/ui-data-mobile.png",
    fullPage: false,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(
    await page
      .locator(".data-page p")
      .first()
      .evaluate((e) => parseFloat(getComputedStyle(e).fontSize)),
  ).toBeGreaterThanOrEqual(14);
  await page
    .getByRole("button", { name: "전체 다운로드 준비", exact: true })
    .click();
  await expect(
    page.getByRole("region", { name: "다운로드 미리보기" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
