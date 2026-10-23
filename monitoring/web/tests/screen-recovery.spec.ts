import { test, expect } from "@playwright/test";

for (const extension of ["js", "css"]) {
  test(`missing viewer ${extension} keeps navigation and refresh recovers MRI`, async ({
    page,
  }) => {
    let missing = true;
    const chunk = new RegExp(`/assets/ViewerPage-[^/]+\\.${extension}$`);
    await page.route(chunk, (route) =>
      missing
        ? route.fulfill({
            status: 404,
            contentType: "text/plain",
            body: "missing build fixture",
          })
        : route.continue(),
    );
    await page.goto("/#data");
    await page.getByRole("button", { name: "영상 분석", exact: true }).click();
    await expect(
      page.getByRole("heading", { name: "영상 분석을 불러오지 못했습니다" }),
    ).toBeVisible();
    await expect(page.locator(".sidebar")).toBeVisible();
    await page
      .getByRole("button", { name: "다운로드·캐시", exact: true })
      .click();
    await expect(page.locator(".raw-data-panel")).toBeVisible();
    await page.getByRole("button", { name: "영상 분석", exact: true }).click();
    await expect(
      page.getByRole("button", { name: "화면 새로고침" }),
    ).toBeVisible();
    await page.screenshot({
      path: `../../.local/ui-viewer-recovery-${extension}.png`,
    });
    missing = false;
    await page.getByRole("button", { name: "화면 새로고침" }).click();
    await expect(
      page.getByRole("heading", { name: "Study 탐색" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "목록 보기", exact: true }).click();
    await page.locator(".viewer-study-list > button").first().click();
    await expect(page.getByLabel("MRI 영상 캔버스")).toBeVisible();
    await expect(page.locator(".viewer-loading")).toHaveCount(0);
    await expect(page.locator(".viewer-stage-label")).toContainText("×");
  });
}

test("persistent load failure stays actionable without reload loop or losing raw selection", async ({
  page,
}) => {
  let navigations = 0;
  page.on("request", (r) => {
    if (r.isNavigationRequest() && r.frame() === page.mainFrame())
      navigations++;
  });
  await page.route(/\/assets\/ViewerPage-[^/]+\.js$/, (r) =>
    r.fulfill({ status: 404, body: "offline fixture" }),
  );
  await page.goto("/?raw=c6cbac7f5974259d656301e1#viewer");
  const recovery = page.getByRole("button", { name: "화면 새로고침" });
  await expect(recovery).toBeVisible();
  await recovery.click();
  await expect(recovery).toBeVisible();
  await page
    .getByRole("button", { name: "다운로드·캐시", exact: true })
    .click();
  await expect(page.locator(".raw-data-panel")).toBeVisible();
  expect(navigations).toBe(2);
  expect(new URL(page.url()).searchParams.get("raw")).toBe(
    "c6cbac7f5974259d656301e1",
  );
});
