import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "path";
import { readFileSync } from "fs";

const packageJson = JSON.parse(readFileSync("./package.json", "utf-8")) as {
  version: string;
};

export default defineConfig({
  base: "/graphmind/",
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
    // API docs URL for the footer; can be overridden via environment variable
    API_DOCS_URL: JSON.stringify(process.env.API_DOCS_URL || "https://graphmind.maleldil.com/docs"),
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        search: resolve(__dirname, "pages/search.html"),
        // "paper-network": resolve(__dirname, "pages/paper-network.html"),
        detail: resolve(__dirname, "pages/detail.html"),
        "abstract-detail": resolve(__dirname, "pages/abstract-detail.html"),
      },
    },
  },
});
