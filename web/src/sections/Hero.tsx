import { motion } from "framer-motion";
import { useJson } from "../data/useJson";
import type { Manifest } from "../data/types";

function CountUp({ value, decimals = 0 }: { value: number; decimals?: number }) {
  // lightweight: framer-motion animate would need a hook; a CSS-free reveal is
  // enough here since the number is short. Show final value with a rise-in.
  return <span>{value.toFixed(decimals)}</span>;
}

function StatCard({
  label,
  value,
  decimals,
  span = "",
  hint,
}: {
  label: string;
  value: number;
  decimals?: number;
  span?: string;
  hint?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className={`card flex flex-col justify-between p-6 ${span}`}
    >
      <p className="eyebrow">{label}</p>
      <div className="mt-6">
        <p className="stat-num">
          <CountUp value={value} decimals={decimals} />
        </p>
        {hint && <p className="mt-1 text-sm text-faint">{hint}</p>}
      </div>
    </motion.div>
  );
}

export default function Hero() {
  const { data } = useJson<Manifest>("manifest.json");
  const h = data?.headline;

  return (
    <header className="hero-glow relative overflow-hidden">
      <div className="mx-auto max-w-content px-6 pb-16 pt-24 md:pt-32">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <p className="eyebrow mb-4">ETT · Master's Thesis</p>
          <h1 className="max-w-4xl text-4xl font-semibold leading-[1.1] tracking-tight text-ink md:text-6xl">
            Forecasting transformer oil temperature — and detecting anomalies from
            its <span className="text-accent">residuals</span>.
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-relaxed text-muted">
            A leakage-free empirical study on the ETT datasets: do complex
            deep-learning models beat simple linear baselines, and do their
            forecast residuals carry a usable anomaly signal?
          </p>
        </motion.div>

        {/* bento metric grid */}
        <div className="mt-14 grid grid-cols-2 gap-4 md:grid-cols-4">
          <StatCard label="Best MAE" value={h?.best_mae ?? 0} decimals={3} span="md:col-span-2 md:row-span-1" hint="lowest oil-temp MAE across all configs" />
          <StatCard label="Models" value={h?.num_models ?? 0} hint="naive → Transformer" />
          <StatCard label="Anomaly types" value={h?.num_anomaly_types ?? 0} hint="spike · level-shift · frozen" />
          <StatCard label="Datasets" value={h?.num_datasets ?? 0} hint="ETTh1 · ETTh2" />
          <StatCard label="Horizons" value={h?.num_horizons ?? 0} hint="24 · 48 · 96 steps" />
          <StatCard label="Seeds / config" value={h?.num_seeds ?? 0} hint="multi-seed robustness" />
        </div>
      </div>
    </header>
  );
}
