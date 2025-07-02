import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "path";
import { readFileSync } from "fs";

const packageJson = JSON.parse(readFileSync("./package.json", "utf-8")) as {
  version: string;
};

export default defineConfig({
  base: "/paper-hypergraph/",
  root: ".",
  publicDir: "public",
  resolve: {
    alias: {
      "@": "/src",
    },
  },
  plugins: [tailwindcss()],
  define: {
    VERSION: JSON.stringify(packageJson.version),
    BUILD_TIME: JSON.stringify(new Date().toISOString()),
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        search: resolve(__dirname, "pages/search.html"),
        // "paper-network": resolve(__dirname, "pages/paper-network.html"),
        detail: resolve(__dirname, "pages/detail.html"),
        "partial-detail": resolve(__dirname, "pages/partial-detail.html"),
      },
    },
  },
});
