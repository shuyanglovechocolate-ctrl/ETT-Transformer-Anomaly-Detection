import { useMemo } from "react";
import Section from "../components/Section";
import { useJson, assetUrl } from "../data/useJson";
import type { AttentionData } from "../data/types";

export default function AttentionAnalysis() {
  const { data } = useJson<AttentionData>("attention.json");
  const layers = data?.layers ?? [];
  const inputLen = data?.input_len ?? 96;

  const derived = useMemo(() => {
    if (!layers.length) return null;
    const avg = (f: (l: AttentionData["layers"][number]) => number) =>
      layers.reduce((s, l) => s + f(l), 0) / layers.length;
    const entropyRatio = avg((l) => l.entropy / l.max_entropy); // ~1 → diffuse
    const recent8 = avg((l) => l.recent_8_mass);
    const uniformRecent = 8 / inputLen; // expected recent-8 mass under uniform attention
    const recencyFactor = recent8 / uniformRecent;
    const peakShare = avg((l) => l.peak_attention); // weight held by the single peak lag
    return { entropyRatio, recent8, recencyFactor, peakShare };
  }, [layers, inputLen]);

  return (
    <Section
      id="attention"
      eyebrow="Supplementary · interpretability"
      title="Where does the Transformer attend?"
      lead="A descriptive look at how the encoder distributes attention over the 96 input lags. This is context for the earlier results, not a driver of them."
      tint
    >
      <div className="grid gap-6 lg:grid-cols-3">
        {/* main image: attention mass by lag (static, no CSV exists for it) */}
        <figure className="card overflow-hidden lg:col-span-2">
          {data && (
            <img
              src={assetUrl(data.figures.by_lag)}
              alt="Mean attention weight over input lags, per encoder layer"
              className="w-full bg-white"
              loading="lazy"
            />
          )}
          <figcaption className="border-t border-border px-5 py-3 text-xs text-faint">
            Mean attention weight per input lag (0 = most recent). Both encoder layers peak near the
            recent steps, then spread thinly across older lags with weak periodic bumps.
          </figcaption>
        </figure>

        {/* dynamic summary card from attention_summary.csv */}
        <div className="card flex flex-col p-6">
          <p className="eyebrow mb-4">Summary</p>

          <div className="overflow-hidden rounded-lg border border-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-surface-2 text-faint">
                  <th className="px-3 py-2 text-left font-medium">Layer</th>
                  <th className="px-3 py-2 text-right font-medium">Peak lag</th>
                  <th className="px-3 py-2 text-right font-medium">Recent-8</th>
                </tr>
              </thead>
              <tbody className="font-mono text-ink">
                {layers.map((l) => (
                  <tr key={l.layer} className="border-t border-border/70">
                    <td className="px-3 py-2 text-left">{l.layer}</td>
                    <td className="px-3 py-2 text-right">{l.peak_lag}</td>
                    <td className="px-3 py-2 text-right">{(l.recent_8_mass * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 space-y-3">
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-muted">Attention entropy</span>
              <span className="font-mono text-sm text-ink">
                {derived ? `${Math.round(derived.entropyRatio * 100)}% of max` : "—"}
              </span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-muted">Peak-lag share</span>
              <span className="font-mono text-sm text-ink">
                {derived ? `${(derived.peakShare * 100).toFixed(1)}%` : "—"}
              </span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-muted">Recency vs uniform</span>
              <span className="font-mono text-sm text-ink">
                {derived ? `${derived.recencyFactor.toFixed(1)}×` : "—"}
              </span>
            </div>
          </div>

          <div className="mt-5 rounded-xl border border-border bg-surface-2 p-4">
            <p className="text-sm leading-relaxed text-ink">
              Attention is <b>diffuse</b>: entropy sits near its maximum and no single lag holds more
              than ~{derived ? (derived.peakShare * 100).toFixed(0) : 4}% of the weight. There is a{" "}
              <b>mild recency bias</b> — the most recent 8 steps carry about{" "}
              {derived ? derived.recencyFactor.toFixed(1) : 3}× their uniform share — but most weight
              spreads across older lags.
            </p>
          </div>
        </div>
      </div>

      {/* representative heatmap + the causal caveat */}
      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <figure className="card overflow-hidden lg:col-span-2">
          {data && (
            <img
              src={assetUrl(data.figures.heatmap)}
              alt="Last-layer attention matrix for a representative window"
              className="w-full bg-white"
              loading="lazy"
            />
          )}
          <figcaption className="border-t border-border px-5 py-3 text-xs text-faint">
            Representative last-layer attention matrix ({data?.experiment_id ?? "Transformer run"}).
          </figcaption>
        </figure>

        <aside className="flex items-center">
          <p className="text-sm italic leading-relaxed text-muted">
            The attention maps provide a descriptive view of where the Transformer allocates weight;
            they do not establish causal feature importance.
          </p>
        </aside>
      </div>
    </Section>
  );
}
