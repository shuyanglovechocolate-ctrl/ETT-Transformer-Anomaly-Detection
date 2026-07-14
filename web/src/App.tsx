import { useState } from "react";
import Hero from "./sections/Hero";
import EDA from "./sections/EDA";
import MethodologyAudit from "./sections/MethodologyAudit";
import ModelComparison from "./sections/ModelComparison";
import ForecastViz from "./sections/ForecastViz";
import InputLengthAblation from "./sections/InputLengthAblation";
import ETTmValidity from "./sections/ETTmValidity";
import AnomalyDetection from "./sections/AnomalyDetection";
import AnomalyStressTest from "./sections/AnomalyStressTest";
import ThresholdExplorer from "./sections/ThresholdExplorer";
import FrozenFailure from "./sections/FrozenFailure";
import AccuracyVsDetection from "./sections/AccuracyVsDetection";
import SignificancePanel from "./sections/SignificancePanel";
import EfficiencyComplexity from "./sections/EfficiencyComplexity";
import LatencyCost from "./sections/LatencyCost";
import AttentionAnalysis from "./sections/AttentionAnalysis";
import DownloadCenter from "./sections/DownloadCenter";
import Citation from "./sections/Citation";
import { useJson } from "./data/useJson";
import type { Manifest } from "./data/types";

// Five thematic groups mirror the report narrative:
// Data & Protocol → Forecasting Evidence → Robustness & Failure →
// Statistical & Practical Evidence → Interpretation & Reproducibility.
// Desktop shows the five group entries; mobile expands each into its sections.
const NAV_GROUPS = [
  {
    label: "Data",
    id: "eda",
    items: [
      { id: "eda", label: "Data" },
      { id: "methodology", label: "Methodology" },
    ],
  },
  {
    label: "Forecasting",
    id: "forecasting",
    items: [
      { id: "forecasting", label: "Model comparison" },
      { id: "forecast-viz", label: "Predictions" },
      { id: "input-length", label: "Input length" },
      { id: "ettm", label: "ETTm validity" },
    ],
  },
  {
    label: "Anomaly",
    id: "anomaly",
    items: [
      { id: "anomaly", label: "Detection" },
      { id: "stress-test", label: "Stress test" },
      { id: "threshold", label: "Threshold" },
      { id: "frozen", label: "Frozen failure" },
    ],
  },
  {
    label: "Evidence",
    id: "accuracy-detection",
    items: [
      { id: "accuracy-detection", label: "Accuracy vs detection" },
      { id: "significance", label: "Significance" },
      { id: "efficiency", label: "Efficiency" },
      { id: "latency", label: "Latency" },
    ],
  },
  {
    label: "Resources",
    id: "attention",
    items: [
      { id: "attention", label: "Attention" },
      { id: "downloads", label: "Downloads" },
      { id: "reproducibility", label: "Reproducibility" },
    ],
  },
];

const GITHUB_URL =
  "https://github.com/shuyanglovechocolate-ctrl/ETT-Transformer-Anomaly-Detection";

function Nav() {
  const [open, setOpen] = useState(false); // mobile menu
  const [openGroup, setOpenGroup] = useState<string | null>(null); // mobile accordion
  const closeMenu = () => {
    setOpen(false);
    setOpenGroup(null);
  };
  return (
    <nav className="sticky top-0 z-50 border-b border-border/60 bg-bg/80 backdrop-blur">
      <div className="mx-auto flex max-w-content items-center justify-between px-6 py-4">
        <a href="#top" className="font-mono text-sm font-medium tracking-tight text-ink">
          ETT<span className="text-accent">/</span>residual-ad
        </a>

        {/* desktop: five group entries, each jumping to the group's first section */}
        <div className="mx-4 hidden min-w-0 flex-1 items-center justify-center gap-8 lg:flex">
          {NAV_GROUPS.map((g) => (
            <a
              key={g.id}
              href={`#${g.id}`}
              className="whitespace-nowrap text-sm text-muted transition-colors hover:text-ink"
            >
              {g.label}
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

      {/* mobile dropdown — five expandable groups */}
      {open && (
        <div id="mobile-nav" className="border-t border-border/60 bg-bg px-4 py-1 lg:hidden">
          <ul>
            {NAV_GROUPS.map((g) => {
              const isOpen = openGroup === g.id;
              return (
                <li key={g.id} className="border-b border-border/40 last:border-0">
                  <button
                    type="button"
                    onClick={() => setOpenGroup(isOpen ? null : g.id)}
                    aria-expanded={isOpen}
                    className="flex w-full items-center justify-between px-2 py-3.5 text-left text-sm font-medium text-ink"
                  >
                    {g.label}
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 16 16"
                      fill="none"
                      aria-hidden="true"
                      className={`transition-transform ${isOpen ? "rotate-180" : ""}`}
                    >
                      <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                  {isOpen && (
                    <ul className="pb-2">
                      {g.items.map((it) => (
                        <li key={it.id}>
                          <a
                            href={`#${it.id}`}
                            onClick={closeMenu}
                            className="block rounded-md px-4 py-2.5 text-sm text-muted transition-colors hover:bg-surface-2 hover:text-ink"
                          >
                            {it.label}
                          </a>
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              );
            })}
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
    <footer id="reproducibility" className="border-t border-border/60 py-16">
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
        {/* Data & Protocol */}
        <Hero />
        <EDA />
        <MethodologyAudit />
        {/* Forecasting Evidence */}
        <ModelComparison />
        <ForecastViz />
        <InputLengthAblation />
        <ETTmValidity />
        {/* Robustness & Failure Analysis */}
        <AnomalyDetection />
        <AnomalyStressTest />
        <ThresholdExplorer />
        <FrozenFailure />
        {/* Statistical & Practical Evidence */}
        <AccuracyVsDetection />
        <SignificancePanel />
        <EfficiencyComplexity />
        <LatencyCost />
        {/* Interpretation & Reproducibility */}
        <AttentionAnalysis />
        <DownloadCenter />
      </main>
      <Footer />
      <Citation />
    </div>
  );
}
