import { test, expect } from "@playwright/test";

test("real cache viewer renders MRI, navigates, measures ROI and saves a review fixture", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#viewer");
  await page.getByRole("button", { name: "목록 보기", exact: true }).click();
  const first = page.locator(".viewer-study-list > button").first();
  await expect(first).toBeEnabled();
  const openedResponse = page.waitForResponse(
    (r) => r.url().endsWith("/api/viewer/open") && r.status() === 200,
  );
  await first.click();
  const opened = await (await openedResponse).json();
  const canvas = page.getByLabel("MRI 영상 캔버스");
  await expect(canvas).toBeVisible();
  await expect(page.locator(".viewer-loading")).toHaveCount(0);
  await expect(page.locator(".viewer-stage-label")).toContainText("288 × 288");
  await expect(page.locator(".viewer-slice-controls")).toContainText("5 / 9");
  await canvas.press("ArrowRight");
  await expect(page.locator(".viewer-slice-controls")).toContainText("6 / 9");
  await expect(page.locator(".viewer-loading")).toHaveCount(0);
  await page.getByLabel("흑백 반전", { exact: true }).check();
  await expect(page.locator(".viewer-loading")).toHaveCount(0);
  await page.getByRole("button", { name: "영역 선택", exact: true }).click();
  await canvas.scrollIntoViewIfNeeded();
  const box = (await canvas.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.45, box.y + box.height * 0.45);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, {
    steps: 8,
  });
  await page.mouse.up();
  await expect(
    page.getByRole("heading", { name: "선택 영역 통계" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "선택 영역 통계" }),
  ).toBeVisible();
  await expect(page.locator(".viewer-loading")).toHaveCount(0);
  await page.getByRole("button", { name: "슬라이스 모아보기" }).click();
  await expect(page.locator(".viewer-contact img")).toHaveCount(9);
  await page
    .getByRole("button", { name: "미리보기 슬라이스 1", exact: true })
    .click();
  await expect(page.locator(".viewer-slice-controls")).toContainText("1 / 9");
  await page.route("**/api/viewer/reviews", async (route) => {
    const body = route.request().postDataJSON();
    expect(body.slice).toBe(0);
    expect(body.roi).toHaveLength(4);
    await route.fulfill({
      json: {
        id: "ui-review-fixture",
        revision: opened.revision,
        slot: 0,
        slice: 0,
        series: "AX_FS",
        note: body.note,
        flags: body.flags,
        frame: { roi: { box: body.roi }, window: [0, 255], invert: true },
      },
    });
  });
  await page
    .getByLabel("영상 검토 메모")
    .fill("UI fixture: ROI and crop check");
  await page.getByLabel("잘림 확인", { exact: true }).check();
  await page.getByRole("button", { name: "검토 기록 저장" }).click();
  await expect(
    page.getByText("이 슬라이스의 검토 기록을 저장했습니다."),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: "검토 JSON" })).toHaveAttribute(
    "href",
    /api\/viewer\/export/,
  );
  await page.getByRole("button", { name: "영역 해제" }).click();
  await page.getByLabel("흑백 반전", { exact: true }).uncheck();
  await expect(page.locator(".viewer-loading")).toHaveCount(0);
  await canvas.scrollIntoViewIfNeeded();
  await page.screenshot({
    path: "../../.local/ui-viewer-desktop.png",
    fullPage: false,
  });
  expect(errors).toEqual([]);
});

test("quality and positive-label filters, missing masks and file browsing", async ({
  page,
}) => {
  await page.goto("/#viewer");
  await page.getByRole("button", { name: "목록 보기", exact: true }).click();
  await expect(
    page.locator(".viewer-study-list > button").first(),
  ).toBeEnabled();
  await page.getByLabel("영상 품질 필터").selectOption("labeled");
  await expect(page.locator(".viewer-list-caption")).toContainText("58");
  await page.getByLabel("양성 소견 필터").selectOption("ACL");
  await expect(page.locator(".viewer-list-caption")).toContainText("24");
  await page.locator(".viewer-study-list > button").first().click();
  await expect(
    page.locator(".viewer-labels .positive").filter({ hasText: "ACL" }),
  ).toContainText("양성");
  await page.getByLabel("양성 소견 필터").selectOption("");
  await page.getByLabel("영상 품질 필터").selectOption("missing");
  await expect(page.locator(".viewer-list-caption")).toContainText("557");
  const response = page.waitForResponse(
    (r) => r.url().endsWith("/api/viewer/open") && r.status() === 200,
  );
  await page.locator(".viewer-study-list > button").first().click();
  const opened = await (await response).json();
  const slot = opened.series.findIndex((s: any) =>
    s.valid.some((v: boolean) => !v),
  );
  await page
    .getByRole("group", { name: "MRI 시리즈" })
    .getByRole("button")
    .nth(slot)
    .click();
  const missing = opened.series[slot].valid.indexOf(false);
  await page
    .getByRole("button", { name: `슬라이스 ${missing + 1}: 결측`, exact: true })
    .click();
  await expect(page.locator(".viewer-missing")).toBeVisible();
  await expect(
    page
      .locator(".viewer-panel .viewer-warning")
      .filter({ hasText: "결측 마스크" }),
  ).toContainText("결측 마스크");
  await page.locator(".viewer-file-panel summary").click();
  await page
    .getByLabel("탐색할 데이터 폴더")
    .selectOption({ label: "캐시 폴더" });
  await expect(page.locator(".viewer-file-browser li")).not.toHaveCount(0);
});

test("empty local installation and invalid file errors are understandable", async ({
  page,
}) => {
  await page.route("**/api/viewer/catalog", (r) =>
    r.fulfill({ json: { caches: [], roots: [] } }),
  );
  await page.goto("/#viewer");
  await page.getByRole("button", { name: "목록 보기", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "확인할 study를 선택하세요" }),
  ).toBeVisible();
  await page.locator(".viewer-file-panel summary").click();
  await page
    .getByLabel("이 PC의 파일 또는 study 폴더 경로")
    .fill("storage/missing-viewer-test.npy");
  await page.getByRole("button", { name: "열기", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText(
    "파일 또는 폴더가 없습니다",
  );
});

test("MRI viewer fits mobile and dark theme without horizontal overflow", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#viewer");
  await page.getByRole("button", { name: "목록 보기", exact: true }).click();
  await expect(
    page.locator(".viewer-study-list > button").first(),
  ).toBeEnabled();
  await page.locator(".viewer-study-list > button").first().click();
  const canvas = page.getByLabel("MRI 영상 캔버스");
  await expect(canvas).toBeVisible();
  await expect(page.locator(".viewer-loading")).toHaveCount(0);
  await canvas.scrollIntoViewIfNeeded();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "../../.local/ui-viewer-mobile.png",
    fullPage: false,
  });
  // Test appearance only; never changes the user's saved preference.
  await page.evaluate(() =>
    document.documentElement.setAttribute("data-theme", "dark"),
  );
  await page.screenshot({
    path: "../../.local/ui-viewer-dark.png",
    fullPage: false,
  });
});
