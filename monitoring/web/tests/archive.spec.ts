import { test, expect } from "@playwright/test";

test("archive structure, descriptions, study pagination, lossless train MRI and mobile", async ({
  page,
}) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#data");
  const panel = page.locator(".raw-data-panel");
  await expect(panel).toContainText("4,410 study");
  await expect(page.getByLabel("원본 폴더 구조")).toContainText(
    "학습 4,407 study",
  );
  await expect(panel).toContainText("Fluid_Sensitive");
  await page.getByLabel("원본 데이터 구분").selectOption("train");
  await expect(panel.locator("article")).toHaveCount(20);
  const uid1 = await panel
    .locator("article")
    .first()
    .locator("code")
    .first()
    .textContent();
  await page.getByRole("button", { name: "다음 study", exact: true }).click();
  await expect(
    panel.locator("article").first().locator("code").first(),
  ).not.toHaveText(uid1!);
  await page.getByRole("button", { name: "이전 study", exact: true }).click();
  const article = panel.locator("article").first();
  await article
    .getByText("시리즈·파일 구조 및 촬영 설명", { exact: true })
    .click();
  await expect(
    article.locator(".raw-series-buttons > button").first(),
  ).toContainText("수분");
  await article.locator(".raw-series-buttons > button").first().click();
  await expect(article.getByLabel("시리즈 파일 목록")).toContainText(".dcm");
  await panel.screenshot({ path: "../../.local/ui-archive-structure.png" });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page
        .locator(".sidebar")
        .evaluate((el) => el.getBoundingClientRect().right),
    )
    .toBeLessThanOrEqual(1);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  await page.screenshot({ path: "../../.local/ui-archive-mobile.png" });
  await page.setViewportSize({ width: 1440, height: 1080 });
  const response = page.waitForResponse(
    (r) => r.url().endsWith("/api/viewer/open") && r.status() === 200,
  );
  await article
    .getByRole("link", { name: /원본 준비 후 열기|영상 열기/ })
    .click();
  const opened = await (await response).json();
  expect(opened.kind).toBe("dicom");
  expect(opened.local_inventory.training_allowed).toBe(true);
  expect(opened.local_inventory.validation).toContain("CRC");
  expect(opened.series.length).toBeGreaterThan(1);
  await expect(page.getByLabel("MRI 영상 캔버스")).toBeVisible();
  await expect(page.locator(".viewer-loading")).toHaveCount(0);
  await expect(page.locator(".viewer-stage-label")).toContainText("×");
  await page.getByRole("button", { name: "슬라이스 모아보기" }).click();
  await expect(page.locator(".viewer-contact img").first()).toBeVisible();
  await page.getByLabel("MRI 영상 캔버스").scrollIntoViewIfNeeded();
  await page.screenshot({ path: "../../.local/ui-archive-train-viewer.png" });
  expect(errors).toEqual([]);
});
