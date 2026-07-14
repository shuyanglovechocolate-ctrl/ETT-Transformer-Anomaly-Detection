import { useState } from "react";
import Hero from "./sections/Hero";
import EDA from "./sections/EDA";
import MethodologyAudit from "./sections/MethodologyAudit";
import ModelComparison from "./sections/ModelComparison";
import ForecastViz from "./sections/ForecastViz";
import InputLengthAblation from "./sections/InputLengthAblation";
import ETTmValidity from "./sections/ETTmValidity";
import SignificancePanel from "./sections/SignificancePanel";
import AnomalyDetection from "./sections/AnomalyDetection";
import AnomalyStressTest from "./sections/AnomalyStressTest";
import ThresholdExplorer from "./sections/ThresholdExplorer";
import AccuracyVsDetection from "./sections/AccuracyVsDetection";
import EfficiencyComplexity from "./sections/EfficiencyComplexity";
import LatencyCost from "./sections/LatencyCost";
import FrozenFailure from "./sections/FrozenFailure";
import AttentionAnalysis from "./sections/AttentionAnalysis";
import DownloadCenter from "./sections/DownloadCenter";
import Citation from "./sections/Citation";
import { useJson } from "./data/useJson";
import type { Manifest } from "./data/types";

const NAV = [
  { id: "eda", label: "Data" },
  { id: "methodology", label: "Methodology" },
  { id: "forecasting", label: "Forecasting" },
  { id: "forecast-viz", label: "Predictions" },
  { id: "input-length", label: "Input length" },
  { id: "ettm", label: "ETTm" },
  { id: "significance", label: "Significance" },
  { id: "anomaly", label: "Anomaly" },
  { id: "stress-test", label: "Stress test" },
  { id: "threshold", label: "Threshold" },
  { id: "accuracy-detection", label: "Acc. vs Detection" },
  { id: "efficiency", label: "Efficiency" },
  { id: "latency", label: "Latency" },
  { id: "frozen", label: "Frozen" },
  { id: "attention", label: "Attention" },
  { id: "downloads", label: "Downloads" },
];

const GITHUB_URL =
  "https://github.com/shuyanglovechocolate-ctrl/ETT-Transformer-Anomaly-Detection";

function Nav() {
  const [open, setOpen] = useState(false);
  return (
    <nav className="sticky top-0 z-50 border-b border-border/60 bg-bg/80 backdrop-blur">
      <div className="mx-auto flex max-w-content items-center justify-between px-6 py-4">
        <a href="#top" className="font-mono text-sm font-medium tracking-tight text-ink">
          ETT<span className="text-accent">/</span>residual-ad
        </a>

        {/* desktop links — horizontally scrollable so the growing section list
            never breaks the layout */}
        <div className="no-scrollbar mx-4 hidden min-w-0 flex-1 gap-6 overflow-x-auto lg:flex">
          {NAV.map((n) => (
            <a
              key={n.id}
              href={`#${n.id}`}
              className="whitespace-nowrap text-sm text-muted transition-colors hover:text-ink"
            >
              {n.label}
            </a>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="rounded-lg border border-border px-3 py-2 text-sm text-muted transition-colors hover:border-accent/50 hover:text-ink"
          >
            GitHub ↗
          </a>
          {/* mobile menu toggle */}
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-controls="mobile-nav"
            aria-label="Toggle section menu"
            className="rounded-lg border border-border p-2 text-ink lg:hidden"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              {open ? (
                <path d="M4 4l10 10M14 4L4 14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              ) : (
                <path d="M2.5 5h13M2.5 9h13M2.5 13h13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* mobile dropdown */}
      {open && (
        <div id="mobile-nav" className="border-t border-border/60 bg-bg px-6 py-2 lg:hidden">
          <ul className="grid grid-cols-2 gap-1 py-2">
            {NAV.map((n) => (
              <li key={n.id}>
                <a
                  href={`#${n.id}`}
                  onClick={() => setOpen(false)}
                  className="block rounded-md px-3 py-2.5 text-sm text-muted transition-colors hover:bg-surface-2 hover:text-ink"
                >
                  {n.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
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
        <EDA />
        <MethodologyAudit />
        <ModelComparison />
        <ForecastViz />
        <InputLengthAblation />
        <ETTmValidity />
        <SignificancePanel />
        <AnomalyDetection />
        <AnomalyStressTest />
        <ThresholdExplorer />
        <AccuracyVsDetection />
        <EfficiencyComplexity />
        <LatencyCost />
        <FrozenFailure />
        <AttentionAnalysis />
        <DownloadCenter />
      </main>
      <Footer />
      <Citation />
    </div>
  );
}
