import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import Segmented from "../components/Segmented";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { EfficiencyRow } from "../data/types";
import { palette, modelColors, modelLabels } from "../theme";

// Trained models that carry parameters (Naive has 0 → shown as a baseline line).
const TRAINED = ["linear", "nlinear", "dlinear", "lstm", "transformer"];

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

export default function EfficiencyComplexity() {
  const { data } = useJson<EfficiencyRow[]>("efficiency.json");
  const [dataset, setDataset] = useState("ETTh1");
  const [inputType, setInputType] = useState<"multivariate" | "univariate">("multivariate");
  const [horizon, setHorizon] = useState(24);

  const rows = useMemo(
    () =>
      (data ?? []).filter(
        (r) => r.dataset === dataset && r.input_type === inputType && r.horizon === horizon
      ),
    [data, dataset, inputType, horizon]
  );

  const naive = rows.find((r) => r.model === "naive");
  const trained = useMemo(
    () => rows.filter((r) => r.params > 0).sort((a, b) => a.params - b.params),
    [rows]
  );

  const stats = useMemo(() => {
    if (!rows.length) return null;
    const bestAcc = [...rows].sort((a, b) => a.mae - b.mae)[0];
    const smallest = trained[0];
    const transformer = rows.find((r) => r.model === "transformer");

    // model that dominates the Transformer: fewer params AND <= MAE, pick lowest MAE
    let verdict: string | null = null;
    if (transformer) {
      const dominators = trained
        .filter((r) => r.params < transformer.params && r.mae <= transformer.mae)
        .sort((a, b) => a.mae - b.mae);
      if (dominators.length) {
        const d = dominators[0];
        const ratio = (transformer.params / d.params).toFixed(1);
        verdict = `${modelLabels[d.model]} reaches a lower MAE than the Transformer with ${ratio}× fewer parameters.`;
      }
    }

    return { bestAcc, smallest, transformer, verdict };
  }, [rows, trained]);

  const option: EChartsOption = useMemo(() => {
    const series = TRAINED.map((m) => ({
      name: modelLabels[m],
      type: "scatter" as const,
      data: trained
        .filter((r) => r.model === m)
        .map((r) => [r.params, r.mae, r.checkpoint_mb ?? 0]),
      symbolSize: 15,
      itemStyle: { color: modelColors[m], opacity: 0.9, borderColor: palette.pointBorder, borderWidth: 1 },
      ...(m === "transformer" && naive
        ? {
            markLine: {
              symbol: "none",
              silent: true,
              lineStyle: { color: palette.faint, type: "dashed" as const, width: 1 },
              label: {
                formatter: `Naive baseline (0 params) · MAE ${naive.mae.toFixed(2)}`,
                color: palette.faint,
                fontSize: 11,
                position: "insideEndTop" as const,
              },
              data: [{ yAxis: naive.mae }],
            },
          }
        : {}),
    }));

    return {
      grid: { left: 10, right: 24, top: 40, bottom: 52, containLabel: true },
      legend: {
        data: TRAINED.map((m) => modelLabels[m]),
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
          const [params, mae, ckpt] = p.value as number[];
          return `<b>${p.seriesName}</b><br/>Parameters ${params.toLocaleString()}<br/>MAE ${mae.toFixed(
            3
          )}<br/>Checkpoint ${ckpt ? ckpt.toFixed(2) + " MB" : "—"}`;
        },
      },
      xAxis: {
        type: "log",
        name: "Parameters  (log scale)  →  (larger model)",
        nameLocation: "middle",
        nameGap: 32,
        nameTextStyle: { color: palette.faint, fontSize: 12 },
        splitLine: { lineStyle: { color: palette.grid } },
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: {
          color: palette.muted,
          formatter: (v: number) => (v >= 1000 ? `${v / 1000}k` : `${v}`),
        },
      },
      yAxis: {
        type: "value",
        name: "MAE  →  (worse)",
        nameLocation: "middle",
        nameGap: 40,
        nameTextStyle: { color: palette.faint, fontSize: 12 },
        scale: true,
        splitLine: { lineStyle: { color: palette.grid } },
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      series,
    };
  }, [trained, naive]);

  return (
    <Section
      id="efficiency"
      title="Does model complexity buy accuracy?"
      lead="Oil-temperature MAE against parameter count (log scale). The dashed line is the zero-parameter Naive baseline. Points below and to the left are the sweet spot: accurate and small."
      tint
    >
      <div className="mb-8 flex flex-wrap gap-x-8 gap-y-4">
        <Segmented label="Dataset" value={dataset} options={["ETTh1", "ETTh2"]} onChange={setDataset} />
        <Segmented
          label="Input"
          value={inputType}
          options={["multivariate", "univariate"]}
          onChange={setInputType}
          format={(v) => (v === "multivariate" ? "Multivariate" : "Univariate")}
        />
        <Segmented label="Horizon" value={horizon} options={[24, 48, 96]} onChange={setHorizon} format={(v) => `${v}h`} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          {trained.length ? (
            <EChart option={option} height={420} />
          ) : (
            <div className="flex h-[420px] items-center justify-center text-faint">
              No trained models for this configuration.
            </div>
          )}
        </div>

        <div className="card flex flex-col p-6">
          <p className="eyebrow mb-2">Verdict</p>
          <StatRow
            label="Best accuracy"
            value={stats ? modelLabels[stats.bestAcc.model] ?? stats.bestAcc.model : "—"}
            hint={stats ? `MAE ${stats.bestAcc.mae.toFixed(3)}` : ""}
          />
          <StatRow
            label="Smallest trained"
            value={stats?.smallest ? modelLabels[stats.smallest.model] : "—"}
            hint={stats?.smallest ? `${stats.smallest.params.toLocaleString()} params` : ""}
          />
          <StatRow
            label="Transformer"
            value={stats?.transformer ? stats.transformer.params.toLocaleString() : "—"}
            hint={stats?.transformer ? `params · MAE ${stats.transformer.mae.toFixed(3)}` : ""}
          />

          <div className="mt-5 rounded-xl bg-accent/[0.06] p-4">
            <p className="text-sm font-medium leading-relaxed text-ink">
              More parameters did not buy accuracy — the strongest linear baseline
              matches or beats the Transformer at a fraction of its size.
            </p>
            {stats?.verdict && (
              <p className="mt-2 text-sm leading-relaxed text-muted">{stats.verdict}</p>
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}
