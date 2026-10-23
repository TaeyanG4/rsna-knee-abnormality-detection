import { test, expect } from "@playwright/test";

test("cache cards identify private project notebooks separately from official data and research", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#data");
  const cards = page.locator(".data-cache-item");
  // Adjacent, Orthodiffusion, and the locally built Raptor training corpus.
  await expect(cards).toHaveCount(3);
  await expect(
    page.getByText("자체 생성 · 비공식 캐시", { exact: true }),
  ).toHaveCount(3);
  const raptor = cards.filter({ hasText: "raptor64_336_c140_s02" });
  await expect(raptor).toHaveCount(1);
  await expect(raptor).toContainText("training/src/raptor_corpus.py");
  await expect(raptor.getByRole("link", { name: "생성 노트북 바로가기" })).toHaveCount(0);
  await expect(cards.first()).toContainText("생성 계정 taeyangg4");
  await expect(
    cards.first().getByRole("link", { name: "생성 노트북 바로가기" }),
  ).toHaveAttribute(
    "href",
    "https://www.kaggle.com/code/taeyangg4/rsna-knee-exp034-adjacent-cache-4x9x288",
  );
  for (const index of [1]) {
    await expect(
      cards.nth(index).getByRole("link", { name: "생성 노트북 바로가기" }),
    ).toHaveAttribute(
      "href",
      "https://www.kaggle.com/code/taeyangg4/rsna-knee-exp032-orthodiffusion-cache-3x16x256",
    );
    await expect(
      cards
        .nth(index)
        .getByRole("link", { name: "OrthoDiffusion 참고 연구 코드" }),
    ).toHaveAttribute(
      "href",
      "https://github.com/lt-0123/OrthoDiffusion/tree/99d15186bb8face1728263023d8414d9405c8f7e",
    );
  }
  await expect(cards.first()).toContainText("비공개 노트북");
  await cards.first().screenshot({ path: "../../.local/ui-cache-origin.png" });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect
    .poll(() =>
      page
        .locator(".sidebar")
        .evaluate((el) => el.getBoundingClientRect().right),
    )
    .toBeLessThanOrEqual(1);
  await cards.nth(1).scrollIntoViewIfNeeded();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  await page.screenshot({ path: "../../.local/ui-cache-origin-mobile.png" });
  expect(errors).toEqual([]);
});

test("unrecognized caches do not inherit a creator from their name", async ({
  page,
}) => {
  await page.route("**/api/data", async (route) => {
    const original = await route.fetch();
    const body = await original.json();
    body.caches = [
      {
        ...body.caches[0],
        manifest_sha256: "unregistered",
        name: "EXP034_renamed_by_someone",
      },
    ];
    await route.fulfill({ json: body });
  });
  await page.goto("/#data");
  await expect(page.locator(".data-cache-item")).toContainText("출처 미확인");
  await expect(
    page.getByRole("link", { name: "생성 노트북 바로가기" }),
  ).toHaveCount(0);
});
