# ETT Showcase Site

A static, interactive showcase for the ETT forecasting + residual anomaly
detection thesis. **Apple-inspired academic** design — light, restrained, generous
whitespace — in service of an academic presentation rather than as a copy of it.
No backend: a Python build step turns the thesis `results/` artifacts into compact
JSON, so the site hosts free on GitHub Pages and stays fast.

**Live:** <https://shuyanglovechocolate-ctrl.github.io/ETT-Transformer-Anomaly-Detection/>

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Build | **Vite 5** | Fast dev server + tree-shaken production bundles |
| UI | **React 18 + TypeScript** (strict) | Component model, fully typed data contracts |
| Styling | **Tailwind CSS 3** + CSS-variable design tokens | One place to retheme; utility-first consistency |
| Charts | **Apache ECharts 5** (on-demand imports) | Interactive canvas charts; only used modules bundled |
| Motion | **Framer Motion** | Scroll-reveal that respects `prefers-reduced-motion` |
| Type | System font stack (SF Pro on macOS) + JetBrains Mono | Native, no webfont for body; mono for figures |
| Data | **Pure-stdlib Python** (`scripts/build_data.py`) | `results/` CSV/JSON → compact static JSON, no deps |
| Hosting | **GitHub Pages** via **GitHub Actions** | Zero backend, zero cost, reproducible deploys |

## Architecture & engineering decisions

- **Static-first, zero backend.** Thesis results are precomputed into JSON at build
  time; the client only fetches static files. This is what makes free GitHub Pages
  hosting possible and keeps every panel fast.
- **Design tokens.** All colours/roles are CSS variables in `src/index.css`,
  mirrored for ECharts in `src/theme.ts`. Re-theming (dark → light → accent) is a
  single-file change; components never hard-code colours.
- **Reusable primitives.** `Section` (scroll-reveal wrapper), `EChart` (thin ECharts
  wrapper), `Segmented` (filter control) and the `useJson` hook keep sections small
  and consistent. Data shapes are typed in `src/data/types.ts`.
- **Accessibility (WCAG 2.1 AA).** Body/caption contrast verified ≥ 4.5:1 on white
  and tint; unified `:focus-visible` ring; `prefers-reduced-motion` fallback (CSS +
  `MotionConfig reducedMotion="user"`); semantic landmarks and heading order; alt
  text and figure/figcaption for images; charts pair colour with position/size.
- **Performance.** ECharts is imported per-module (`echarts/core` + registered
  charts), cutting the bundle from ~1.31 MB to ~0.87 MB (gzip 433 → 287 KB); images
  are lazy-loaded; chart options are memoised.
- **Responsive.** Mobile-first; a hamburger menu below `lg`, a horizontally
  scrollable section nav above it, and filter controls with ~44 px touch targets.
- **Reproducible data.** CI regenerates `public/data/` from committed `results/`
  tables on every deploy, so the site can never drift from the thesis numbers.
  (Heavy source CSVs that are git-ignored have their small derived JSON committed.)

## Prerequisites

- Node.js **≥ 18** (use `nvm install 20 && nvm use 20`)
- Python 3 (standard library only — the data script needs no `pip install`)

## Develop

```bash
cd web
npm run data     # regenerate public/data/*.json (+ figures) from ../results
npm install
npm run dev      # http://localhost:5173
```

## Build

```bash
npm run build    # type-check (tsc) + production bundle into web/dist
npm run preview  # serve the production build locally
```

## Deploy

Pushing to `main` (touching `web/**` or `results/**`) triggers
`.github/workflows/deploy-web.yml`, which rebuilds the data, builds the site and
publishes to GitHub Pages. Enable Pages → "GitHub Actions" once in repo settings.

The production base path is `/ETT-Transformer-Anomaly-Detection/` (set in
`vite.config.ts`). Override with the `VITE_BASE` env var if the repo is renamed or
served from a custom domain.

## Data pipeline

`scripts/build_data.py` (pure standard library) reads committed `results/` tables
and writes `public/data/`:

| Source | Output |
|---|---|
| `metrics/model_comparison.csv` | `comparison.json` |
| `predictions/*.csv` (downsampled first-step series) | `predictions/*.json` |
| `anomaly/metrics/*.csv` | `anomaly.json` |
| `metrics/efficiency_complexity_summary.csv` | `efficiency.json` |
| `external_validity/ettm_forecasting_comparison.csv` | `ettm.json` |
| `sensitivity/input_len_ablation*.csv` | `input_length.json` |
| `anomaly/metrics/*` (frozen diagnostics/hybrid) | `frozen.json` |
| `metrics/attention_summary.csv` (+ copied figures) | `attention.json` |
| `metrics/reproducibility_manifest.json` | `manifest.json` |

Generated `public/data/` is git-ignored and regenerated in CI (except the
prediction JSON, whose source CSVs are too heavy to commit, so the small JSON are
committed and preserved when the CSVs are absent).

## Sections (narrative order)

1. **Hero** — headline metrics (bento grid)
2. **EDA** — dataset background: OT trend, feature series, correlation structure
3. **Forecasting** — interactive MAE comparison across models / datasets / horizons
4. **Predictions** — actual vs. predicted oil temperature with zoom
5. **Input length** — MAE/RMSE/WAPE across look-back 48 / 96 / 192
6. **ETTm external validity** — does the linear-family advantage transfer to 15-min data?
7. **Anomaly** — residual detector performance by anomaly type
8. **Accuracy vs. detection** — forecasting accuracy vs. detection quality scatter
9. **Efficiency** — accuracy vs. model complexity (parameters / checkpoint size)
10. **Frozen failure** — the residual blind spot and the temporal-flatness fix
11. **Attention** — exploratory Transformer attention over input lags

A closing **Citation** block (APA + BibTeX + copy) completes the page.
