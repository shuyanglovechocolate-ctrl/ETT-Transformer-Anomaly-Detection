import Section from "../components/Section";
import { palette } from "../theme";

// The strict, never-reordered processing order (README Module 1 & 4). Reordering
// any step would open a leakage path, so it is stated as a fixed contract.
const PIPELINE = [
  "Raw ETT series",
  "Chronological split · 70 / 10 / 20",
  "Fit scaler on train only",
  "Transform train / val / test",
  "Window independently per split",
];

// Each row: a concrete leakage risk and the guardrail that closes it. Every
// claim is grounded in the thesis protocol, not invented for the site.
const GUARDRAILS: { risk: string; guard: string }[] = [
  {
    risk: "Shuffling the series would let future rows train the model.",
    guard: "Splits are strictly chronological — the data is never shuffled.",
  },
  {
    risk: "Fitting the scaler on all data leaks the val/test distribution into training.",
    guard: "StandardScaler is fit on the training split only, then applied unchanged to val and test.",
  },
  {
    risk: "Windowing before splitting creates samples that straddle a split boundary.",
    guard: "Sliding windows are built independently inside each split; boundary-crossing windows are dropped.",
  },
  {
    risk: "Reporting metrics in scaled space is hard to interpret and easy to game.",
    guard: "Predictions, ground truth and residuals are inverse-transformed to the original OT scale (scaler_y).",
  },
  {
    risk: "Tuning the anomaly threshold on the test set inflates detection scores.",
    guard: "Thresholds are learned only from validation residuals; the test split is scored once.",
  },
  {
    risk: "Re-training on injected anomalies would let the detector peek at the labels.",
    guard: "Anomalies are injected into y_true only — forecasts and model weights stay untouched.",
  },
  {
    risk: "A single lucky seed can manufacture a winner.",
    guard: "Every comparison runs three seeds (42 / 2024 / 3407) with paired-bootstrap confidence intervals.",
  },
];

function Check() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true" className="mt-0.5 shrink-0">
      <circle cx="8" cy="8" r="8" fill={palette.accent} fillOpacity="0.12" />
      <path d="M4.5 8.2l2.2 2.2 4.8-4.8" stroke={palette.accent} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function MethodologyAudit() {
  return (
    <Section
      id="methodology"
      eyebrow="Protocol · leakage audit"
      title="Every number comes from one leakage-free protocol"
      lead="Forecasting accuracy and the residual anomaly signal are only meaningful if no future information reaches the model. The pipeline is built so that each common leakage path is closed by construction — here is the audit."
    >
      <div className="card p-6">
        <p className="eyebrow mb-4">Processing order · never reordered</p>
        <ol className="flex flex-col gap-2 lg:flex-row lg:flex-wrap lg:items-stretch">
          {PIPELINE.map((step, i) => (
            <li key={step} className="flex items-center gap-2">
              <span className="flex min-h-[44px] flex-1 items-center rounded-lg border border-border bg-surface px-4 py-2.5 text-sm text-ink lg:flex-none">
                <span className="mr-2.5 font-mono text-xs text-faint">{i + 1}</span>
                {step}
              </span>
              {i < PIPELINE.length - 1 && (
                <span className="hidden text-faint lg:mx-0.5 lg:inline" aria-hidden="true">
                  →
                </span>
              )}
            </li>
          ))}
        </ol>
        <p className="mt-4 text-sm leading-relaxed text-muted">
          Fitting the scaler on the full series, or building windows before splitting, would each expose
          future data to training. Doing both in this order — and only this order — keeps the train set
          blind to everything that comes after it.
        </p>
      </div>

      <div className="mt-6 grid gap-3 md:grid-cols-2">
        {GUARDRAILS.map((g) => (
          <div key={g.risk} className="card flex flex-col gap-3 p-5">
            <div className="flex items-start gap-2.5">
              <span
                className="mt-1 h-2 w-2 shrink-0 rounded-full"
                style={{ background: palette.danger }}
                aria-hidden="true"
              />
              <p className="text-sm leading-relaxed text-muted">
                <span className="mr-1.5 font-mono text-[11px] uppercase tracking-wider text-faint">Risk</span>
                {g.risk}
              </p>
            </div>
            <div className="flex items-start gap-2.5 border-t border-border/60 pt-3">
              <Check />
              <p className="text-sm leading-relaxed text-ink">
                <span className="mr-1.5 font-mono text-[11px] uppercase tracking-wider" style={{ color: palette.accent }}>
                  Guarded
                </span>
                {g.guard}
              </p>
            </div>
          </div>
        ))}
      </div>

      <p className="mt-6 max-w-2xl text-sm leading-relaxed text-muted">
        The same protocol carries into the anomaly study: Module 4 reuses the frozen forecast residuals
        without retraining, learns thresholds on validation only, and compares the residual detector against
        three causal statistical baselines under an identical evaluation. Nothing downstream ever sees the
        test split more than once.
      </p>
    </Section>
  );
}
