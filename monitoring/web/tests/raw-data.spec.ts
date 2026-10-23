import { test, expect } from "@playwright/test";

test("downloaded raw study opens from the data page without changing downloads", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#data");
  const panel = page.locator(".raw-data-panel");
  await expect(panel).toContainText("전체 원본 연결됨 · 4,410 study");
  await expect(panel).toContainText("DICOM 819,635개");
  await expect(panel.locator("article")).toHaveCount(20);
  await expect(panel).toContainText("전체 데이터 구조·설명");
  await page.getByLabel("원본 데이터 구분").selectOption("test");
  await expect(panel.locator("article")).toHaveCount(3);
  const exported = await page.request.get("/api/data/library/descriptor");
  expect(
    (await exported.json()).raw
      .filter((row: any) => row.split === "test")
      .every((row: any) => !row.training_allowed),
  ).toBe(true);
  await panel.screenshot({ path: "../../.local/ui-raw-data-panel.png" });
  await panel.getByRole("link", { name: "영상 열기" }).first().click();
  await page.getByRole("button", { name: "목록 보기", exact: true }).click();
  await expect(page.getByLabel("영상 캐시")).toHaveValue("raw");
  await expect(page.locator(".viewer-stage-label")).toContainText("×");
  await expect(
    page.locator(".viewer-workspace > .viewer-warning"),
  ).toContainText("공개 테스트");
  await expect(
    page.locator(".viewer-workspace > .viewer-warning"),
  ).not.toContainText("부분 다운로드");
  await expect(
    page.getByRole("group", { name: "MRI 시리즈" }).getByRole("button"),
  ).toHaveCount(5);
  await page
    .getByRole("group", { name: "MRI 시리즈" })
    .getByRole("button")
    .nth(1)
    .click();
  await expect(page.locator(".viewer-stage-label")).toContainText("×");
  await expect(page.locator(".viewer-loading")).toHaveCount(0);
  await page.screenshot({
    path: "../../.local/ui-raw-dicom.png",
    fullPage: false,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#data");
  await expect(panel).toContainText("DICOM 819,635개");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});
