// Shared palette for ECharts (which cannot read CSS variables directly).
// Keep in sync with src/index.css design tokens (light / Apple-inspired).
export const palette = {
  bg: "#ffffff",
  surface: "#ffffff",
  border: "#d6d6db",
  ink: "#1d1d1f",
  muted: "#5f6066", // AA on white & tint
  faint: "#6e6e73", // AA (≥4.5:1) on white & tint
  accent: "#006ad6", // Apple blue, AA for small text on tint
  accent2: "#ff9500", // Apple orange
  grid: "#e8e8ed", // light gridlines
  danger: "#d93a3a", // failure / frozen highlight
  pointBorder: "#ffffff", // scatter point outline for separation
  zoomFill: "rgba(0,106,214,0.10)", // dataZoom selected band
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
