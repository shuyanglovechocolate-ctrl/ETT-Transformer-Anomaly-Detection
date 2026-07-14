import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import Segmented from "../components/Segmented";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { AnomalyData } from "../data/types";
import { palette, anomalyColors } from "../theme";

type StressRow = AnomalyData["stress_test"][number];
type MetricKey = "pr_auc" | "roc_auc" | "best_f1" | "event_recall" | "detection_delay";

const METRICS: {
  key: MetricKey;
  label: string;
  higherBetter: boolean;
  digits: number;
  blurb: string;
}[] = [
  { key: "pr_auc", label: "PR-AUC", higherBetter: true, digits: 3, blurb: "threshold-free precision–recall area" },
  { key: "roc_auc", label: "ROC-AUC", higherBetter: true, digits: 3, blurb: "threshold-free ranking quality" },
  { key: "best_f1", label: "Best F1", higherBetter: true, digits: 3, blurb: "oracle F1 at the ideal threshold" },
  { key: "event_recall", label: "Event recall", higherBetter: true, digits: 2, blurb: "fraction of anomaly events caught" },
  { key: "detection_delay", label: "Detection delay", higherBetter: false, digits: 1, blurb: "steps until first alarm (lower is better)" },
];

// The six injected anomaly shapes: three original + three added in the stress test.
const TYPES = ["spike", "level_shift", "frozen", "drift", "noise_burst", "stuck_with_jitter"];
const TYPE_LABEL: Record<string, string> = {
  spike: "Spike",
  level_shift: "Level shift",
  frozen: "Frozen",
  drift: "Drift",
  noise_burst: "Noise burst",
  stuck_with_jitter: "Stuck + jitter",
};

const DETECTORS = ["residual", "diff_score", "raw_zscore", "rolling_zscore", "flatness"];
const DET_LABEL: Record<string, string> = {
  residual: "Residual",
  diff_score: "Diff score",
  raw_zscore: "Raw z-score",
  rolling_zscore: "Rolling z-score",
  flatness: "Flatness",
};

export default function AnomalyStressTest() {
  const { data } = useJson<AnomalyData>("anomaly.json");
  const [metric, setMetric] = useState<MetricKey>("pr_auc");

  const rows = data?.stress_test ?? [];
  const mag = data?.magnitude_sensitivity ?? [];
  const meta = METRICS.find((m) => m.key === metric)!;

  // (anomaly_type -> detector -> row) lookup for O(1) cell access.
  const lookup = useMemo(() => {
    const m: Record<string, Record<string, StressRow>> = {};
    for (const r of rows) (m[r.anomaly_type] ??= {})[r.detector_type] = r;
    return m;
  }, [rows]);

  // Range of the selected metric across every populated cell, for color scaling.
  const { min, max } = useMemo(() => {
    const vals = rows.map((r) => r[metric]).filter((v): v is number => v != null);
    return vals.length ? { min: Math.min(...vals), max: Math.max(...vals) } : { min: 0, max: 1 };
  }, [rows, metric]);

  // Best detector per anomaly type on the selected metric (skips empty cells).
  const bestByType = useMemo(() => {
    const out: Record<string, { detector: string; value: number } | null> = {};
    for (const t of TYPES) {
      let best: { detector: string; value: number } | null = null;
      for (const d of DETECTORS) {
        const v = lookup[t]?.[d]?.[metric];
        if (v == null) continue;
        if (!best || (meta.higherBetter ? v > best.value : v < best.value)) best = { detector: d, value: v };
      }
      out[t] = best;
    }
    return out;
  }, [lookup, metric, meta.higherBetter]);

  // 0..1 quality of a cell (inverted for delay so "good" always reads darker).
  const quality = (v: number) => {
    if (max === min) return 0.5;
    const t = (v - min) / (max - min);
    return meta.higherBetter ? t : 1 - t;
  };

  const magOption: EChartsOption = useMemo(() => {
    const types = [...new Set(mag.map((r) => r.anomaly_type))];
    const scales = [...new Set(mag.map((r) => r.magnitude_scale))].sort((a, b) => a - b);
    return {
      grid: { left: 8, right: 20, top: 30, bottom: 8, containLabel: true },
      legend: { data: types.map((t) => TYPE_LABEL[t] ?? t), top: 0, textStyle: { color: palette.muted } },
      tooltip: { trigger: "axis", confine: true, backgroundColor: palette.surface, borderColor: palette.border, textStyle: { color: palette.ink } },
      xAxis: {
        type: "category",
        data: scales.map((s) => `${s}×`),
        name: "magnitude scale",
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: { color: palette.faint, fontSize: 12 },
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      yAxis: { type: "value", max: 1, name: "F1", nameTextStyle: { color: palette.faint }, splitLine: { lineStyle: { color: palette.grid } }, axisLabel: { color: palette.muted } },
      series: types.map((t) => ({
        name: TYPE_LABEL[t] ?? t,
        type: "line" as const,
        data: scales.map((s) => mag.find((r) => r.anomaly_type === t && r.magnitude_scale === s)?.f1 ?? null),
        symbol: "circle",
        symbolSize: 8,
        lineStyle: { width: 2.2, color: anomalyColors[t] },
        itemStyle: { color: anomalyColors[t] },
      })),
    };
  }, [mag]);

  const gridCols = { gridTemplateColumns: "minmax(96px, 1.1fr) repeat(5, minmax(0, 1fr))" };

  return (
    <Section
      id="stress-test"
      eyebrow="Module 4 · Detector stress test"
      title="Six anomaly shapes against five detectors"
      lead="Beyond the residual score, the same injected series are scored by five detectors across six anomaly shapes. Switch the metric to see who wins where — and where the honest gaps are: ROC-AUC was only measured on the original three shapes, and event recall / detection delay only on the three added ones."
    >
      <div className="mb-8">
        <Segmented
          label="Metric"
          value={metric}
          options={METRICS.map((m) => m.key)}
          onChange={(v) => setMetric(v as MetricKey)}
          format={(v) => METRICS.find((m) => m.key === v)?.label ?? String(v)}
        />
        <p className="mt-2 font-mono text-xs text-faint">{meta.blurb}</p>
      </div>

      {rows.length === 0 ? (
        <div className="card flex h-[320px] items-center justify-center text-faint">Loading…</div>
      ) : (
        <>
          <div className="card overflow-x-auto p-4 md:p-5">
            <div className="min-w-[560px]">
              {/* header row */}
              <div className="grid gap-1.5" style={gridCols}>
                <div />
                {DETECTORS.map((d) => (
                  <div key={d} className="px-1 pb-2 text-center font-mono text-[11px] uppercase tracking-wider text-faint">
                    {DET_LABEL[d]}
                  </div>
                ))}
              </div>
              {/* one row per anomaly type */}
              {TYPES.map((t) => (
                <div key={t} className="mb-1.5 grid items-stretch gap-1.5" style={gridCols}>
                  <div className="flex items-center text-sm font-medium text-ink">{TYPE_LABEL[t]}</div>
                  {DETECTORS.map((d) => {
                    const v = lookup[t]?.[d]?.[metric];
                    const isBest = bestByType[t]?.detector === d && v != null;
                    if (v == null) {
                      return (
                        <div
                          key={d}
                          className="flex h-11 items-center justify-center rounded-md border border-dashed border-border/70 text-sm text-faint"
                          title={`${TYPE_LABEL[t]} · ${DET_LABEL[d]} · ${meta.label} not measured`}
                        >
                          —
                        </div>
                      );
                    }
                    return (
                      <div
                        key={d}
                        className="relative flex h-11 items-center justify-center rounded-md text-sm tabular-nums text-ink"
                        style={{
                          background: `rgba(0, 106, 214, ${(0.08 + 0.48 * quality(v)).toFixed(3)})`,
                          boxShadow: isBest ? `inset 0 0 0 2px ${palette.accent}` : undefined,
                          fontWeight: isBest ? 600 : 400,
                        }}
                        title={`${TYPE_LABEL[t]} · ${DET_LABEL[d]} · ${meta.label} ${v.toFixed(meta.digits)}${
                          isBest ? " (best)" : ""
                        }`}
                      >
                        {isBest && (
                          <span
                            className="absolute left-1 top-0.5 text-[10px] leading-none"
                            style={{ color: palette.accent }}
                            aria-label="best detector"
                          >
                            ★
                          </span>
                        )}
                        {v.toFixed(meta.digits)}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1.5 px-1 text-xs text-faint">
            <span>
              <span style={{ color: palette.accent }}>★</span> best detector for the row
            </span>
            <span>
              <span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm border border-dashed border-border/70 align-middle" />
              not measured
            </span>
            <span>deeper fill = {meta.higherBetter ? "higher" : "lower"} {meta.label} (better)</span>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-2">
            {/* best signal per type */}
            <div className="card flex flex-col p-6">
              <p className="eyebrow mb-4">Best detection signal · {meta.label}</p>
              <ul className="space-y-2.5">
                {TYPES.map((t) => {
                  const b = bestByType[t];
                  return (
                    <li key={t} className="flex items-baseline justify-between gap-3 text-sm">
                      <span className="text-muted">{TYPE_LABEL[t]}</span>
                      {b ? (
                        <span className="text-ink">
                          <b>{DET_LABEL[b.detector]}</b>
                          <span className="ml-2 font-mono text-faint">{b.value.toFixed(meta.digits)}</span>
                        </span>
                      ) : (
                        <span className="font-mono text-faint">not measured</span>
                      )}
                    </li>
                  );
                })}
              </ul>
              <div className="mt-5 rounded-xl bg-accent/[0.06] p-4">
                <p className="text-sm leading-relaxed text-ink">
                  The residual score is strongest on most shapes, but it is not universal: diff-score edges
                  it out on noise bursts, and the ranking reshuffles as you switch the metric — the z-score
                  detectors, for instance, raise the alarm with far less delay even when their PR-AUC is
                  lower. No single detector is best everywhere; surfacing that trade-off is the point.
                </p>
              </div>
            </div>

            {/* magnitude sensitivity */}
            <div className="card p-5">
              <p className="eyebrow mb-4 px-1">Residual F1 vs. anomaly magnitude</p>
              <EChart option={magOption} height={300} />
              <p className="mt-2 px-1 text-sm leading-relaxed text-muted">
                For the shapes with a graded magnitude, the residual F1 stays essentially flat from 1× to
                5× — detection neither collapses nor improves much as the anomaly grows. The signal is
                stable across injection strengths, so the result is not an artifact of one chosen magnitude.
              </p>
            </div>
          </div>
        </>
      )}
    </Section>
  );
}
