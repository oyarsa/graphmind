import { defineConfig } from "vite";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "path";

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
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        "paper-explorer": resolve(__dirname, "pages/paper-explorer.html"),
        // "paper-network": resolve(__dirname, "pages/paper-network.html"),
        "paper-detail": resolve(__dirname, "pages/paper-detail.html"),
      },
    },
  },
});
