import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "tests",
  timeout: 30000,
  workers: 1,
  use: {
    baseURL: process.env.KNEEOPS_URL || "http://127.0.0.1:8765",
    headless: true,
    viewport: { width: 1440, height: 1080 },
  },
  outputDir: "../../.local/ui-results",
});
