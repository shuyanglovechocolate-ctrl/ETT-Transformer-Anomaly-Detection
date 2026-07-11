import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves a project site under /<repo>/. Set base accordingly for
// production builds; dev server stays at root. Override with VITE_BASE if the
// repo is ever renamed or served from a custom domain.
const base = process.env.VITE_BASE ?? "/ETT-Transformer-Anomaly-Detection/";

export default defineConfig(({ command }) => ({
  base: command === "build" ? base : "/",
  plugins: [react()],
}));
