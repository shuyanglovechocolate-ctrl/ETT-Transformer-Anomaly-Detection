import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import Segmented from "../components/Segmented";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { InputLengthData } from "../data/types";
import { palette, modelColors, modelLabels } from "../theme";

type Metric = "mae" | "rmse" | "wape";
const METRIC_LABEL: Record<Metric, string> = { mae: "MAE", rmse: "RMSE", wape: "WAPE" };
const MODELS = ["nlinear", "dlinear", "transformer"];

export default function InputLengthAblation() {
  const { data } = useJson<InputLengthData>("input_length.json");
  const [metric, setMetric] = useState<Metric>("mae");

  const rows = data?.rows ?? [];
  const summary = data?.summary ?? [];
  const lens = useMemo(() => [...new Set(rows.map((r) => r.input_len))].sort((a, b) => a - b), [rows]);

  // seed-variance contrast (transformer vs linear family) for the callout
  const variance = useMemo(() => {
    const stdOf = (models: string[]) =>
      rows.filter((r) => models.includes(r.model)).map((r) => r.mae_std);
    const t = stdOf(["transformer"]);
    const lin = stdOf(["nlinear", "dlinear"]);
    const fmt = (a: number[]) => (a.length ? `${Math.min(...a).toFixed(2)}–${Math.max(...a).toFixed(2)}` : "—");
    return { transformer: fmt(t), linear: fmt(lin) };
  }, [rows]);

  const option: EChartsOption = useMemo(() => {
    const series = MODELS.map((m) => ({
      name: modelLabels[m] ?? m,
      type: "line" as const,
      data: lens.map((L) => {
        const r = rows.find((x) => x.model === m && x.input_len === L);
        return r ? r[metric] : null;
      }),
      symbol: "circle",
      symbolSize: 8,
      lineStyle: { width: 2.2, color: modelColors[m] },
      itemStyle: { color: modelColors[m] },
    }));
    return {
      grid: { left: 8, right: 24, top: 34, bottom: 8, containLabel: true },
      legend: { data: MODELS.map((m) => modelLabels[m] ?? m), top: 2, textStyle: { color: palette.muted } },
      tooltip: {
        trigger: "axis",
        backgroundColor: palette.surface,
        borderColor: palette.border,
        textStyle: { color: palette.ink },
        formatter: (ps: any) => {
          const L = lens[ps[0].dataIndex];
          const lines = MODELS.map((m) => {
            const r = rows.find((x) => x.model === m && x.input_len === L);
            if (!r) return "";
            return `${modelLabels[m]}: ${METRIC_LABEL[metric]} ${r[metric].toFixed(3)}${
              metric === "mae" ? ` ± ${r.mae_std.toFixed(3)}` : ""
            }`;
          });
          return `<b>input length ${L}</b><br/>${lines.filter(Boolean).join("<br/>")}`;
        },
      },
      xAxis: {
        type: "category",
        data: lens.map(String),
        name: "input length",
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: { color: palette.faint, fontSize: 12 },
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      yAxis: {
        type: "value",
        name: METRIC_LABEL[metric] + (metric === "wape" ? " (%)" : ""),
        nameTextStyle: { color: palette.faint },
        scale: true,
        splitLine: { lineStyle: { color: palette.grid } },
        axisLabel: { color: palette.muted },
      },
      series,
    };
  }, [rows, lens, metric]);

  return (
    <Section
      id="input-length"
      eyebrow="Ablation · input length"
      title="How much does look-back length matter?"
      lead="Re-running the strongest models at input lengths 48 / 96 / 192 (horizon 96). The question: does a longer look-back let the Transformer overtake the linear family? It does not."
    >
      <div className="mb-8 flex flex-wrap gap-x-8 gap-y-4">
        <Segmented
          label="Metric"
          value={metric}
          options={["mae", "rmse", "wape"]}
          onChange={setMetric}
          format={(v) => METRIC_LABEL[v as Metric]}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          {rows.length ? <EChart option={option} height={360} /> : <div className="flex h-[360px] items-center justify-center text-faint">Loading…</div>}
        </div>

        <div className="card flex flex-col p-6">
          <p className="eyebrow mb-4">Best per length</p>
          <ul className="space-y-2">
            {summary.map((s) => (
              <li key={s.input_len} className="flex items-baseline justify-between text-sm">
                <span className="font-mono text-faint">len {s.input_len}</span>
                <span className="text-ink">{s.ranking}</span>
              </li>
            ))}
          </ul>

          <div className="mt-5 rounded-xl bg-accent/[0.06] p-4">
            <p className="text-sm leading-relaxed text-ink">
              The Transformer stays last at every look-back length, and its seed variance is far
              higher (±{variance.transformer} MAE vs ±{variance.linear} for the linear family). The
              linear ranking does shift — DLinear leads at 48, NLinear takes over and keeps improving
              as the window grows.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
}
