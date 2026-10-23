import { test, expect } from "@playwright/test";

const TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"];
const files = {
  predictions: [
    { id: "artifacts/predictions/very_long_prediction_file_name_for_layout_checks_gold58.csv", name: "very_long_prediction_file_name_for_layout_checks_gold58.csv", folder: "artifacts/predictions", bytes: 17000, targets: TARGETS, role: "prediction" },
    { id: "artifacts/experiments/b/pred.csv", name: "pred.csv", folder: "artifacts/experiments/b", bytes: 11000, targets: TARGETS, role: "prediction" },
  ],
  labels: [
    { id: "storage/competition/train.csv", name: "train.csv", folder: "storage/competition", bytes: 1, targets: TARGETS, role: "label" },
    { id: "storage/labels/a_really_long_teacher_label_folder_name_that_used_to_widen_the_select/labels.csv", name: "labels.csv", folder: "storage/labels/x", bytes: 1, targets: TARGETS, role: "label" },
  ],
};
const [a, b] = files.predictions.map((f) => f.id);
const auc = (i: number) => ({ [a]: 0.8 + i / 100, [b]: 0.7 + i / 100, blend: 0.85 + i / 100 });
const result = {
  common_studies: 58,
  files: files.predictions.map((f) => ({ ...f, rows: 58, min: 0, max: 1, outside_probability: false, missing: 0 })),
  correlations: [{ a, b, macro: 0.71, per_target: Object.fromEntries(TARGETS.map((t, i) => [t, 0.5 + i / 30])) }],
  targets: TARGETS,
  warnings: ["공식 라벨은 58 study뿐이며 Gold58은 기존 선택에 노출된 진단셋입니다."],
  blend: "equal-weight rank mean",
  label: {
    id: "storage/competition/train.csv", name: "train.csv", studies: 58, computable_targets: TARGETS,
    per_target: TARGETS.map((t, i) => ({ target: t, label_mode: "binary", n_pos: 10 + i, n_neg: 48 - i, auc: auc(i) })),
    macro: { [a]: 0.86, [b]: 0.76, blend: 0.91 },
    bootstrap: { [a]: { low: 0.8, high: 0.9, valid: 200 }, [b]: { low: 0.7, high: 0.8, valid: 200 }, blend: { low: 0.88, high: 0.94, valid: 200 },
      blend_vs_best: { reference: a, low: 0.01, high: 0.08, fraction_positive: 0.98, unit: "study", reps: 200 } },
  },
};

test("prediction comparison shows guarded diagnostics and keeps its layout", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  let posted: Record<string, any> | null = null;
  await page.route("**/api/predictions/files*", (r) => r.fulfill({ json: files }));
  await page.route("**/api/predictions/compare", (r) => {
    posted = r.request().postDataJSON();
    return r.fulfill({ json: result });
  });
  await page.goto("/#predictions");
  await expect(page.getByRole("heading", { level: 1, name: "실험·제출" })).toBeVisible();
  await expect(page.getByRole("button", { name: "비교 실행" })).toBeDisabled();
  await page.getByText("pred.csv", { exact: true }).click();
  await page.getByText(files.predictions[0].name).click();
  await page.getByRole("button", { name: "비교 실행" }).click();
  await expect(page.locator(".stat-tile")).toHaveCount(3);
  expect(posted).toEqual({ files: [b, a], label: "storage/competition/train.csv" });
  await expect(page.getByText("Gold58은 기존 선택에 노출된 진단셋", { exact: false })).toBeVisible();
  await expect(page.getByRole("table", { name: "소견별 AUC" }).getByRole("row")).toHaveCount(13);
  await expect(page.getByText(/양수 비율 98%/)).toBeVisible();
  // A long label name must not widen the picker over the results.
  const picker = (await page.locator(".prediction-picker").boundingBox())!;
  const select = (await page.locator(".prediction-picker select").boundingBox())!;
  expect(select.x + select.width).toBeLessThanOrEqual(picker.x + picker.width + 1);
  // A phone opens the page at its own width and runs the comparison there.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.reload();
  await page.getByText("pred.csv", { exact: true }).click();
  await page.getByRole("button", { name: "비교 실행" }).click();
  await expect(page.locator(".stat-tile").first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(1);
  expect(errors).toEqual([]);
});

test("library opens notebooks on the current competition with sort and state filters", async ({ page }) => {
  const requests: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/api/sources?")) requests.push(r.url());
  });
  await page.goto("/#notebooks");
  await expect(page.getByLabel("대회 선택")).toHaveValue("rsna-knee-abnormality-detection");
  await expect(page.getByLabel("정렬")).toHaveValue("activity");
  await expect(page.getByRole("columnheader", { name: "추가된 날짜 (KST)" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "마지막 실행" })).toBeVisible();
  await page.getByLabel("정렬").selectOption("votes");
  await page.getByLabel("본문 상태").selectOption("full");
  await expect.poll(() => requests.some((u) => u.includes("sort=votes") && u.includes("body=full"))).toBe(true);
  await expect(page.locator(".state-pill.full").first()).toBeVisible();
  const chips = page.locator(".top-inputs li button");
  if (await chips.count()) {
    await page.locator(".top-inputs summary").click();
    await chips.first().click();
    await expect(page.locator(".filter-chip")).toBeVisible();
    await expect.poll(() => requests.some((u) => /input=[^&]+/.test(u))).toBe(true);
    await page.getByRole("button", { name: "입력 필터 해제" }).click();
    await expect(page.locator(".filter-chip")).toHaveCount(0);
  }
  await page.getByRole("button", { name: /토론/ }).first().click();
  await expect(page.getByLabel("정렬")).toHaveValue("activity");
  await expect(page.getByRole("option", { name: "댓글 많은 순" })).toBeAttached();
});

test("source dialog compares inputs and shows a marked diff", async ({ page }) => {
  await page.route("**/api/source/compare*", (r) =>
    r.fulfill({
      json: {
        a: { id: "x" }, b: { id: "notebook:u/other", title: "Other fork" }, similarity: 0.9, added: 2, removed: 1, truncated: false,
        inputs: { datasets: { only_a: ["w/new-weights"], only_b: [], both: ["w/base"] } },
        diff: "--- a\n+++ b\n@@ -1 +1 @@\n-old line\n+new line\n+another",
      },
    }),
  );
  await page.goto("/#notebooks");
  await page.getByLabel("본문 상태").selectOption("full");
  await page.locator(".source-title button").first().click();
  const dialog = page.locator("dialog[open]");
  await expect(dialog).toBeVisible();
  await dialog.locator(".compare-panel-source summary").click();
  await dialog.getByLabel("비교할 자료 검색").fill("rsna");
  await dialog.locator(".compare-options button").first().click();
  await expect(dialog.getByText("w/new-weights")).toBeVisible();
  await dialog.getByRole("button", { name: /줄 단위 차이 보기/ }).click();
  await expect(dialog.locator(".diff-view .diff-add")).toHaveCount(2);
  await expect(dialog.locator(".diff-view .diff-del")).toHaveCount(1);
  await dialog.getByRole("button", { name: /차이 닫고 본문 보기/ }).click();
  await expect(dialog.locator(".diff-view")).toHaveCount(0);
});
