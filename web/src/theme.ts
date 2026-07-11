// Shared palette for ECharts (which cannot read CSS variables directly).
// Keep in sync with src/index.css design tokens (light / Apple-inspired).
export const palette = {
  bg: "#ffffff",
  surface: "#ffffff",
  border: "#d6d6db",
  ink: "#1d1d1f",
  muted: "#6e6e73",
  faint: "#86868b",
  accent: "#0071e3", // Apple blue
  accent2: "#ff9500", // Apple orange
  grid: "#e8e8ed", // light gridlines
};

// Stable color per forecasting model, tuned for contrast on a white canvas.
export const modelColors: Record<string, string> = {
  naive: "#8e8e93",
  linear: "#5a6b7b",
  nlinear: "#30a0a8",
  dlinear: "#34a853",
  lstm: "#9b59d0",
  transformer: "#0071e3",
};

export const anomalyColors: Record<string, string> = {
  spike: "#0071e3",
  level_shift: "#ff9500",
  frozen: "#d93a3a",
};

export const modelLabels: Record<string, string> = {
  naive: "Naive",
  linear: "Linear",
  nlinear: "NLinear",
  dlinear: "DLinear",
  lstm: "LSTM",
  transformer: "Transformer",
};
