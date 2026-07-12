import Section from "../components/Section";
import { assetUrl } from "../data/useJson";

function Thumb({
  src,
  alt,
  caption,
  className = "",
  height = 208,
}: {
  src: string;
  alt: string;
  caption: React.ReactNode;
  className?: string;
  height?: number;
}) {
  return (
    <figure className={`card flex flex-col overflow-hidden ${className}`}>
      <div className="flex flex-1 items-center justify-center bg-white p-2" style={{ minHeight: height }}>
        <img
          src={assetUrl(src)}
          alt={alt}
          className="max-h-full max-w-full object-contain"
          loading="lazy"
        />
      </div>
      <figcaption className="border-t border-border px-4 py-2.5 text-xs leading-snug text-faint">
        {caption}
      </figcaption>
    </figure>
  );
}

const FINDINGS = [
  {
    title: "Non-stationary & seasonal",
    body: "Oil temperature drifts across the two-year span with clear seasonal structure — persistence is a real baseline to beat.",
  },
  {
    title: "Correlated, not causal",
    body: "The load features move together, but visible correlation is not predictive gain; that is settled by the ablation, not the heatmap.",
  },
  {
    title: "Leakage-free protocol",
    body: "Every experiment uses a strictly chronological train/val/test split and a scaler fit on the training set only.",
  },
];

export default function EDA() {
  return (
    <Section
      id="eda"
      eyebrow="Data · exploratory analysis"
      title="The ETT data at a glance"
      lead="Background on the Electricity Transformer Temperature (ETTh1) series before the modelling — what the target looks like, how the seven signals behave, and how leakage is avoided."
    >
      {/* bento grid */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* OT trend — largest, spans full width as a wide banner */}
        <figure className="card overflow-hidden lg:col-span-3">
          <div className="bg-white p-2">
            <img
              src={assetUrl("figures/eda_ot_trend.png")}
              alt="Oil temperature over time (ETTh1)"
              className="w-full object-contain"
              loading="lazy"
            />
          </div>
          <figcaption className="border-t border-border px-4 py-2.5 text-xs text-faint">
            Target signal: oil temperature over ~2 years — clearly non-stationary, with seasonal
            swings and occasional sharp dips.
          </figcaption>
        </figure>

        {/* supporting thumbnails */}
        <Thumb
          src="figures/eda_feature_timeseries.png"
          alt="The seven input features over time"
          height={240}
          caption="The seven input signals (six loads + OT) over the same window."
        />
        <Thumb
          src="figures/eda_ot_distribution.png"
          alt="Distribution of oil temperature"
          height={240}
          caption="Distribution of oil temperature — wide, skewed, far from a tidy Gaussian."
        />
        <Thumb
          src="figures/eda_correlation_heatmap.png"
          alt="Feature correlation heatmap"
          height={240}
          caption={
            <>
              Feature correlations — <span className="text-muted">exploratory only</span>: feature
              value is decided by the univariate vs multivariate ablation, not read off this matrix.
            </>
          }
        />
      </div>

      {/* three takeaways */}
      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        {FINDINGS.map((f) => (
          <div key={f.title} className="card p-5">
            <p className="text-sm font-semibold text-ink">{f.title}</p>
            <p className="mt-2 text-sm leading-relaxed text-muted">{f.body}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}
