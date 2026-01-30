import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Exclude e2e tests from vitest - they run with Playwright
    exclude: ["**/node_modules/**", "**/tests/e2e/**"],
  },
});
