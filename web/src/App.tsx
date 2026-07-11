import Hero from "./sections/Hero";
import ModelComparison from "./sections/ModelComparison";
import ForecastViz from "./sections/ForecastViz";
import AnomalyDetection from "./sections/AnomalyDetection";
import AccuracyVsDetection from "./sections/AccuracyVsDetection";
import EfficiencyComplexity from "./sections/EfficiencyComplexity";
import { useJson } from "./data/useJson";
import type { Manifest } from "./data/types";

const NAV = [
  { id: "forecasting", label: "Forecasting" },
  { id: "forecast-viz", label: "Predictions" },
  { id: "anomaly", label: "Anomaly" },
  { id: "accuracy-detection", label: "Accuracy vs Detection" },
  { id: "efficiency", label: "Efficiency" },
];

function Nav() {
  return (
    <nav className="sticky top-0 z-50 border-b border-border/60 bg-bg/80 backdrop-blur">
      <div className="mx-auto flex max-w-content items-center justify-between px-6 py-4">
        <a href="#top" className="font-mono text-sm font-medium tracking-tight text-ink">
          ETT<span className="text-accent">/</span>residual-ad
        </a>
        <div className="hidden gap-6 md:flex">
          {NAV.map((n) => (
            <a key={n.id} href={`#${n.id}`} className="text-sm text-muted transition-colors hover:text-ink">
              {n.label}
            </a>
          ))}
        </div>
        <a
          href="https://github.com/shuyanglovechocolate-ctrl/ETT-Transformer-Anomaly-Detection"
          target="_blank"
          rel="noreferrer"
          className="rounded-lg border border-border px-3 py-1.5 text-sm text-muted transition-colors hover:border-accent/50 hover:text-ink"
        >
          GitHub ↗
        </a>
      </div>
    </nav>
  );
}

function Footer() {
  const { data } = useJson<Manifest>("manifest.json");
  const repro = data?.reproducibility;
  return (
    <footer className="border-t border-border/60 py-16">
      <div className="mx-auto max-w-content px-6">
        <p className="eyebrow mb-4">Reproducibility</p>
        <div className="grid gap-4 text-sm text-muted md:grid-cols-3">
          <div>
            <p className="text-faint">Environment</p>
            <p className="mt-1 font-mono text-ink">
              Python {repro?.python_version ?? "—"} · torch {repro?.package_versions?.torch ?? "—"}
            </p>
            <p className="mt-1 font-mono">{repro?.compute_device ?? "—"}</p>
          </div>
          <div>
            <p className="text-faint">Commit</p>
            <p className="mt-1 font-mono text-ink">{repro?.git_commit?.slice(0, 10) ?? "—"}</p>
            <p className="mt-1 font-mono">generated {repro?.generated_at?.slice(0, 10) ?? "—"}</p>
          </div>
          <div>
            <p className="text-faint">Note</p>
            <p className="mt-1">
              Static site built from the thesis result artifacts. No SOTA claim — a
              leakage-free study of forecasting accuracy vs. residual anomaly signal.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <div id="top">
      <Nav />
      <main>
        <Hero />
        <ModelComparison />
        <ForecastViz />
        <AnomalyDetection />
        <AccuracyVsDetection />
        <EfficiencyComplexity />
      </main>
      <Footer />
    </div>
  );
}
