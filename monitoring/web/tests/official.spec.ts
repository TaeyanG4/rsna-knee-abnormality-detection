import { test, expect } from "@playwright/test";

test("six main groups keep legacy notebook, solution and submission routes", async ({
  page,
}) => {
  await page.goto("/#own");
  const main = page.getByRole("navigation", { name: "주 메뉴" });
  await expect(main.getByRole("button")).toHaveCount(6);
  await expect(
    page.getByRole("heading", { level: 1, name: "자료 라이브러리" }),
  ).toBeVisible();
  await expect(
    page.locator(".section-tabs button[aria-current=page]"),
  ).toContainText("내 노트북");
  await expect(page.locator(".source-row-meta .badge").first()).toHaveText(
    "내 노트북",
  );
  await page.getByRole("button", { name: /로컬 문서/ }).click();
  await expect(page.locator(".source-row-meta .badge").first()).toHaveText(
    "로컬 문서",
  );
  await main.getByRole("button", { name: "과거 대회", exact: true }).click();
  await page.getByRole("button", { name: /과거 솔루션/ }).click();
  await expect(page.locator(".source-row-meta .badge").first()).toHaveText(
    "솔루션 토론",
  );
  await main.getByRole("button", { name: "실험·제출" }).click();
  await page.getByRole("button", { name: "제출 이력", exact: true }).click();
  await expect(page).toHaveURL(/#submissions$/);
  await page.goBack();
  await expect(
    page.locator(".section-tabs button[aria-current=page]"),
  ).toHaveText("실험 기록");
});

test("official overview and complete rules support English, Korean and comparison", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#official");
  const paper = page.locator(".official-paper");
  await expect(
    paper.getByRole("heading", { name: "대회 개요", exact: true }),
  ).toBeVisible();
  const ko = page.getByRole("region", { name: "한국어 번역 본문" });
  const en = page.getByRole("region", { name: "영어 원문 본문" });
  await expect(ko).toContainText("6억 5,400만 명");
  await page.getByRole("button", { name: "English", exact: true }).click();
  await expect(en).toContainText("654 million");
  await expect(ko).toHaveCount(0);
  await page.getByRole("button", { name: "나란히 보기" }).click();
  await expect(en).toBeVisible();
  await expect(ko).toBeVisible();
  await page.screenshot({
    path: "../../.local/ui-official-bilingual.png",
    fullPage: false,
  });
  await page
    .getByRole("navigation", { name: "공식 문서 목록" })
    .getByRole("button", { name: /대회 규칙/ })
    .click();
  await expect(ko).toContainText("최대 5개");
  await expect(ko).toContainText("최대 2개");
  await expect(
    ko.getByRole("heading", { name: "18. 정의", exact: true }),
  ).toBeVisible();
  await expect(en).toContainText("DEFINITIONS");
  await expect(ko.locator("table")).toHaveCount(2);
  await page.getByRole("button", { name: "한국어", exact: true }).click();
  await expect(en).toHaveCount(0);
  await page
    .getByRole("navigation", { name: "공식 문서 목록" })
    .getByRole("button", { name: /평가 방식/ })
    .click();
  await expect(ko.locator(".katex-display")).toBeVisible();
  expect(errors).toEqual([]);
});

test("changed source falls back to English and never displays the old translation", async ({
  page,
}) => {
  await page.route("**/api/source?**", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.translation = {
      status: "outdated",
      message: "원문이 변경되어 번역을 다시 확인해야 합니다.",
    };
    body.revision.content =
      '# Updated source\nNew official rules.\n\n[unsafe](javascript:alert(1))\n<script>alert(1)</script>\n<img src="https://example.com/tracker.png" />';
    await route.fulfill({ json: body });
  });
  await page.goto("/#official");
  await expect(
    page.getByRole("button", { name: "한국어", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("region", { name: "영어 원문 본문" }),
  ).toContainText("New official rules.");
  await expect(
    page.getByRole("region", { name: "한국어 번역 본문" }),
  ).toHaveCount(0);
  await expect(
    page.locator(
      '.official-prose script, .official-prose img, .official-prose a[href^="javascript:"]',
    ),
  ).toHaveCount(0);
  await expect(page.getByRole("status")).toContainText("번역을 다시 확인");
});

test("official guide stays readable on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#official");
  await expect(
    page.getByRole("region", { name: "한국어 번역 본문" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  expect(
    await page
      .locator(".official-prose")
      .first()
      .evaluate((e) => parseFloat(getComputedStyle(e).fontSize)),
  ).toBeGreaterThanOrEqual(16);
  await page.screenshot({
    path: "../../.local/ui-official-mobile.png",
    fullPage: false,
  });
});
