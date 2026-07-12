import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import Segmented from "../components/Segmented";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { AnomalyData } from "../data/types";
import { palette, modelColors, modelLabels } from "../theme";
import { pearson, spearman, fmtR } from "../lib/stats";

type Row = AnomalyData["accuracy_vs_detection"][number];

const MODEL_ORDER = ["naive", "linear", "nlinear", "dlinear", "lstm", "transformer"];
const ANOMALY_LABEL: Record<string, string> = {
  spike: "Spike",
  level_shift: "Level shift",
  frozen: "Frozen",
};

function StatRow({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-border/70 py-3 last:border-0">
      <span className="text-sm text-muted">{label}</span>
      <span className="text-right">
        <span className="font-mono text-base font-medium text-ink">{value}</span>
        {hint && <span className="ml-2 text-xs text-faint">{hint}</span>}
      </span>
    </div>
  );
}

export default function AccuracyVsDetection() {
  const { data } = useJson<AnomalyData>("anomaly.json");
  const [anomaly, setAnomaly] = useState("spike");
  const [dataset, setDataset] = useState<"All" | "ETTh1" | "ETTh2">("All");

  const rows: Row[] = useMemo(() => {
    const all = data?.accuracy_vs_detection ?? [];
    return all.filter(
      (r) => r.anomaly_type === anomaly && (dataset === "All" || r.dataset === dataset)
    );
  }, [data, anomaly, dataset]);

  // correlation across the filtered points (x = forecast MAE, y = average precision)
  const stats = useMemo(() => {
    if (rows.length < 2) return null;
    const x = rows.map((r) => r.forecast_mae);
    const y = rows.map((r) => r.average_precision);
    const r = pearson(x, y);
    const rho = spearman(x, y);

    // per-model mean average precision → best / worst "detector"
    const byModel = new Map<string, number[]>();
    for (const row of rows) {
      if (!byModel.has(row.model)) byModel.set(row.model, []);
      byModel.get(row.model)!.push(row.average_precision);
    }
    const means = [...byModel.entries()].map(([m, aps]) => ({
      model: m,
      ap: aps.reduce((s, v) => s + v, 0) / aps.length,
    }));
    means.sort((a, b) => b.ap - a.ap);
    const best = means[0];
    const worst = means[means.length - 1];

    let observation: string;
    if (r < -0.1)
      observation =
        "Forecasting error and detection quality are weakly to negatively related — better forecasts do not translate into better detection.";
    else if (r <= 0.1)
      observation =
        "Forecasting error and detection quality show essentially no correlation across models.";
    else
      observation =
        "There is only a mild association — forecasting accuracy explains little of the detection performance.";

    return { r, rho, best, worst, observation, n: rows.length };
  }, [rows]);

  const option: EChartsOption = useMemo(() => {
    const series = MODEL_ORDER.map((m) => {
      const pts = rows
        .filter((r) => r.model === m)
        .map((r) => [r.forecast_mae, r.average_precision, r.event_recall, r.horizon]);
      return {
        name: modelLabels[m],
        type: "scatter" as const,
        data: pts,
        symbolSize: (d: number[]) => 7 + d[2] * 21, // size encodes event recall (0→7px, 1→28px)
        itemStyle: {
          color: modelColors[m],
          opacity: 0.82,
          borderColor: palette.pointBorder,
          borderWidth: 0.6,
        },
      };
    });

    return {
      grid: { left: 10, right: 20, top: 44, bottom: 52, containLabel: true },
      legend: {
        data: MODEL_ORDER.map((m) => modelLabels[m]),
        top: 4,
        textStyle: { color: palette.muted },
        itemWidth: 10,
        itemHeight: 10,
      },
      tooltip: {
        backgroundColor: palette.surface,
        borderColor: palette.border,
        textStyle: { color: palette.ink, fontSize: 12 },
        formatter: (p: any) => {
          const [mae, ap, er, h] = p.value as number[];
          return `<b>${p.seriesName}</b> · h${h}<br/>Forecast MAE ${mae.toFixed(
            2
          )}<br/>Avg precision ${ap.toFixed(3)}<br/>Event recall ${er.toFixed(2)}`;
        },
      },
      xAxis: {
        type: "value",
        name: "Forecast MAE  →  (worse forecast)",
        nameLocation: "middle",
        nameGap: 32,
        nameTextStyle: { color: palette.faint, fontSize: 12 },
        min: 1,
        splitLine: { lineStyle: { color: palette.grid } },
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      yAxis: {
        type: "value",
        name: "Average precision  →  (better detection)",
        nameLocation: "middle",
        nameGap: 40,
        nameTextStyle: { color: palette.faint, fontSize: 12 },
        min: 0,
        max: 1,
        splitLine: { lineStyle: { color: palette.grid } },
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      series,
    };
  }, [rows]);

  return (
    <Section
      id="accuracy-detection"
      eyebrow="Module 4 · Key finding"
      title="Forecast Accuracy vs Detection Capability"
      lead="Each point is one model run: its forecasting error (x) against how well its residuals detect injected anomalies (y). Point size encodes event recall. If accuracy drove detection, points would fall on a downward line — they do not."
    >
      <div className="mb-8 flex flex-wrap gap-x-8 gap-y-4">
        <Segmented
          label="Anomaly"
          value={anomaly}
          options={["spike", "level_shift", "frozen"]}
          onChange={setAnomaly}
          format={(v) => ANOMALY_LABEL[v]}
        />
        <Segmented
          label="Dataset"
          value={dataset}
          options={["All", "ETTh1", "ETTh2"]}
          onChange={setDataset}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          <EChart option={option} height={420} />
          <p className="mt-2 px-1 text-xs text-faint">
            Point size = event recall · color = model · {stats?.n ?? 0} runs shown
          </p>
        </div>

        <div className="card flex flex-col p-6">
          <p className="eyebrow mb-2">Correlation</p>
          <StatRow
            label="Pearson r"
            value={stats ? fmtR(stats.r) : "—"}
            hint="MAE vs AP"
          />
          <StatRow label="Spearman ρ" value={stats ? fmtR(stats.rho) : "—"} hint="rank-based" />
          <StatRow
            label="Best detector"
            value={stats ? modelLabels[stats.best.model] : "—"}
            hint={stats ? `AP ${stats.best.ap.toFixed(2)}` : ""}
          />
          <StatRow
            label="Worst detector"
            value={stats ? modelLabels[stats.worst.model] : "—"}
            hint={stats ? `AP ${stats.worst.ap.toFixed(2)}` : ""}
          />

          <div className="mt-5 rounded-xl bg-accent/[0.06] p-4">
            <p className="text-sm font-medium leading-relaxed text-ink">
              Lower forecasting error does not necessarily imply stronger anomaly
              detection performance.
            </p>
            {stats && (
              <p className="mt-2 text-sm leading-relaxed text-muted">{stats.observation}</p>
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}
