import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for frontend e2e tests.
 *
 * These tests require:
 * - Backend server running on :8000 (just api-dev from repo root)
 * - Frontend dev server running on :5173 (just dev from frontend/)
 *
 * Run with: just e2e
 *
 * Paper evaluation tests run in parallel (2 workers). Note: May occasionally
 * hit API rate limits from arXiv and Semantic Scholar.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  /* Allow parallel test execution within files */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Parallel workers: 3 workers to balance speed vs backend rate limits */
  workers: 3,
  /* Reporter to use */
  reporter: [["list"], ["html", { open: "never" }]],
  /* Shared settings for all the projects below */
  use: {
    /* Base URL to use in actions like `await page.goto('/')` */
    baseURL: "http://localhost:5173/graphmind/",
    /* Collect trace when retrying the failed test */
    trace: "on-first-retry",
    /* Increase timeout for API-heavy operations */
    actionTimeout: 10000,
  },
  /* Configure timeout for tests (evaluations can take 2-3 minutes) */
  timeout: 5 * 60 * 1000,
  /* Configure projects for major browsers - default to chromium only for speed */
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    /* Uncomment for cross-browser testing */
    // {
    //   name: "firefox",
    //   use: { ...devices["Desktop Firefox"] },
    // },
    // {
    //   name: "webkit",
    //   use: { ...devices["Desktop Safari"] },
    // },
  ],
  /* Start both backend and frontend servers before tests */
  webServer: [
    {
      command: "just api-dev",
      cwd: "..",
      url: "http://localhost:8000/health",
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
    {
      command: "just dev",
      url: "http://localhost:5173/graphmind/",
      reuseExistingServer: !process.env.CI,
      timeout: 30000,
    },
  ],
});
