import { test, expect } from "@playwright/test";

test("personal library navigation, list, new markers and source dialog", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { level: 1, name: "대회 안내" }),
  ).toBeVisible();
  await page
    .getByRole("navigation", { name: "주 메뉴" })
    .getByRole("button", { name: /자료 라이브러리/ })
    .click();
  await page
    .getByRole("button", { name: /공개 노트북/ })
    .first()
    .click();
  await expect(
    page.getByRole("heading", { level: 1, name: "자료 라이브러리" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "추가된 날짜 (KST)" }),
  ).toBeVisible();
  await expect(page.locator(".source-table tbody tr").first()).toBeVisible();
  await expect(page.locator(".new-badge").first()).toHaveText("NEW");
  await page.screenshot({
    path: "../../.local/ui-library-desktop.png",
    fullPage: false,
  });
  await page.locator(".source-title button").first().click();
  await expect(page.locator("dialog[open]")).toBeVisible();
  await expect(page.getByLabel("저장된 버전")).toBeVisible();
  await page.getByRole("button", { name: "자료 닫기" }).click();
  await page.getByRole("button", { name: /토론/ }).first().click();
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "자료 라이브러리",
      exact: true,
    }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("readable settings and history", async ({ page }) => {
  await page.goto("/#settings");
  await expect(
    page.getByRole("heading", { name: "수집과 상태" }),
  ).toBeVisible();
  await expect(page.getByRole("switch", { name: /정기 수집/ })).toBeAttached();
  await expect(page.getByLabel("수집 간격")).toBeVisible();
  const labelSize = await page
    .locator(".check-field")
    .first()
    .evaluate((e) => parseFloat(getComputedStyle(e).fontSize));
  expect(labelSize).toBeGreaterThanOrEqual(14);
  await page.screenshot({
    path: "../../.local/ui-settings-desktop.png",
    fullPage: false,
  });
  await page.getByRole("button", { name: "과거 대회", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "RSNA 2024 · Lumbar Spine MRI",
      exact: true,
    }),
  ).toBeVisible();
  await page
    .locator("section", {
      has: page.getByRole("heading", {
        name: "RSNA 2024 · Lumbar Spine MRI",
        exact: true,
      }),
    })
    .getByRole("button", { name: "솔루션 보기" })
    .click();
  await expect(page.locator(".source-table tbody tr").first()).toBeVisible();
});

test("mobile menu and page overflow", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await page.getByRole("button", { name: "메뉴 열기" }).click();
  await page
    .getByRole("navigation", { name: "주 메뉴" })
    .getByRole("button", { name: /자료 라이브러리/ })
    .click();
  await page
    .getByRole("button", { name: /공개 노트북/ })
    .first()
    .click();
  await expect(page.locator(".source-table")).toBeVisible();
  await expect
    .poll(async () => (await page.locator(".sidebar").boundingBox())?.x || 0)
    .toBeLessThan(-240);
  await expect(page.locator(".source-row-meta .badge").first()).toHaveText(
    "공개 노트북",
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "../../.local/ui-library-mobile.png",
    fullPage: false,
  });
});
