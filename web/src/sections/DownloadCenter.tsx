import Section from "../components/Section";
import { dataUrl } from "../data/useJson";

const REPO = "https://github.com/shuyanglovechocolate-ctrl/ETT-Transformer-Anomaly-Detection";

// The processed JSON files that power the charts on this site. Same-origin, so
// the `download` attribute streams them straight to disk. These are the exact
// numbers behind every figure — nothing is hidden in a backend.
const DATASETS: { file: string; label: string; desc: string }[] = [
  { file: "comparison.json", label: "Forecasting comparison", desc: "MAE / RMSE / WAPE per model · dataset · horizon (mean ± std)." },
  { file: "significance.json", label: "Significance tests", desc: "Paired-bootstrap 95% CIs on every pairwise MAE difference." },
  { file: "input_length.json", label: "Input-length ablation", desc: "Accuracy vs look-back window (48 / 96 / 192)." },
  { file: "ettm.json", label: "ETTm external validity", desc: "Cross-dataset check on the minute-level ETTm1 / ETTm2." },
  { file: "efficiency.json", label: "Efficiency vs complexity", desc: "Parameter count, checkpoint size and epochs per model." },
  { file: "latency.json", label: "Inference latency", desc: "Measured runtime cost per model (ms / 1k windows)." },
  { file: "anomaly.json", label: "Anomaly detection", desc: "Detector × type × threshold, stress test and magnitude sweep." },
  { file: "frozen.json", label: "Frozen-failure diagnostics", desc: "Where the residual signal collapses and the flatness fix." },
  { file: "attention.json", label: "Attention analysis", desc: "Per-layer entropy, peak lag and recency mass." },
  { file: "manifest.json", label: "Run manifest", desc: "Environment, seeds, git commit and headline statistics." },
];

function DownloadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true" className="shrink-0">
      <path d="M8 2v8m0 0L5 7m3 3l3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 12.5h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

export default function DownloadCenter() {
  return (
    <Section
      id="downloads"
      eyebrow="Open data · reproducibility"
      title="Take the numbers with you"
      lead="Every figure on this page is drawn from committed result tables — no backend, no hidden aggregation. Download the processed data behind each section, or browse the raw CSV artifacts and code on GitHub."
    >
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {DATASETS.map((d) => (
          <a
            key={d.file}
            href={dataUrl(d.file)}
            download
            className="card group flex flex-col p-5 transition-colors hover:border-accent/50"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="font-medium text-ink">{d.label}</span>
              <span className="flex items-center gap-1.5 text-faint transition-colors group-hover:text-accent">
                <span className="font-mono text-[11px] uppercase tracking-wider">JSON</span>
                <DownloadIcon />
              </span>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-muted">{d.desc}</p>
          </a>
        ))}
      </div>

      <div className="mt-6 card flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="font-medium text-ink">Raw result tables &amp; source code</p>
          <p className="mt-1 text-sm text-muted">
            The original per-run CSVs under <code className="font-mono text-ink">results/</code> and the full
            leakage-free pipeline live in the repository.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <a
            href={`${REPO}/tree/main/results`}
            target="_blank"
            rel="noreferrer"
            className="rounded-lg border border-border px-4 py-2.5 text-sm text-ink transition-colors hover:border-accent/50 hover:text-accent"
          >
            Result CSVs ↗
          </a>
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="rounded-lg border border-border px-4 py-2.5 text-sm text-ink transition-colors hover:border-accent/50 hover:text-accent"
          >
            Repository ↗
          </a>
        </div>
      </div>
    </Section>
  );
}
