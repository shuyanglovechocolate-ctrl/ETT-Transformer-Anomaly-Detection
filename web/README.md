# ETT Showcase Site

A static showcase for the ETT forecasting + residual anomaly detection thesis,
with an Apple-inspired academic design (light, restrained, generous whitespace —
inspired by Apple's clarity, but in service of an academic presentation rather
than a copy of it). Built with Vite + React + TypeScript + Tailwind + Apache
ECharts. No backend: a Python build step turns the thesis `results/` artifacts
into compact JSON that the site fetches at runtime, so it hosts free on GitHub
Pages.

## Prerequisites

- Node.js **≥ 18** (local dev is currently on 16 — run `nvm install 20 && nvm use 20`)
- Python 3 (standard library only — the data script needs no pip installs)

## Develop

```bash
cd web
npm run data     # regenerate public/data/*.json from ../results
npm install
npm run dev      # http://localhost:5173
```

## Build

```bash
npm run build    # type-check + production bundle into web/dist
npm run preview  # serve the production build locally
```

## Deploy

Pushing to `main` (touching `web/**` or `results/**`) triggers
`.github/workflows/deploy-web.yml`, which rebuilds the data, builds the site and
publishes to GitHub Pages. Enable Pages → "GitHub Actions" once in repo settings.

The production base path is `/ETT-Transformer-Anomaly-Detection/` (set in
`vite.config.ts`). Override with the `VITE_BASE` env var if the repo is renamed
or served from a custom domain.

## Data pipeline

`scripts/build_data.py` reads:

- `results/metrics/model_comparison.csv` → `comparison.json`
- `results/predictions/*.csv` → `predictions/*.json` (first-step series, downsampled)
- `results/anomaly/metrics/*.csv` → `anomaly.json`
- `results/metrics/reproducibility_manifest.json` → `manifest.json`

Generated `public/data/` is git-ignored; CI regenerates it on every deploy.

## Sections

1. **Hero** — headline metrics (bento grid)
2. **Forecasting** — interactive MAE comparison across models / datasets / horizons
3. **Predictions** — actual vs. predicted oil temperature with zoom
4. **Anomaly** — residual detector performance by anomaly type, incl. the frozen failure mode

Planned: EDA, accuracy-vs-detection scatter, efficiency, attention analysis.
