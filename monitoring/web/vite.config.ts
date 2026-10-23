import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  // Open tabs still reference their build's hashed lazy chunks. Keep these
  // immutable files across builds; replacing index.html selects the new build.
  build: { outDir: "../kneeops/static", emptyOutDir: false },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (outgoing, incoming) => {
            if (
              ["http://127.0.0.1:5173", "http://localhost:5173"].includes(
                incoming.headers.origin || "",
              )
            ) {
              outgoing.setHeader("Origin", "http://127.0.0.1:8765");
            }
          });
        },
      },
    },
  },
});
