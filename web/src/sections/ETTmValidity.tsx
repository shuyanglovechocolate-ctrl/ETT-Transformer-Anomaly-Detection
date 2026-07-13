import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import Segmented from "../components/Segmented";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { EttmRow } from "../data/types";
import { palette, modelColors, modelLabels } from "../theme";

type Metric = "mae" | "rmse" | "wape";
const METRIC_LABEL: Record<Metric, string> = { mae: "MAE", rmse: "RMSE", wape: "WAPE" };

export default function ETTmValidity() {
  const { data } = useJson<EttmRow[]>("ettm.json");
  const [dataset, setDataset] = useState("ETTm1");
  const [metric, setMetric] = useState<Metric>("mae");

  const rows = useMemo(() => {
    const all = (data ?? []).filter((r) => r.dataset === dataset);
    return [...all].sort((a, b) => a[metric] - b[metric]); // best (lowest) first
  }, [data, dataset, metric]);

  const best = rows[0];
  const transformer = rows.find((r) => r.model === "transformer");

  const option: EChartsOption = useMemo(() => {
    return {
      grid: { left: 8, right: 24, top: 20, bottom: 8, containLabel: true },
      tooltip: {
        trigger: "axis",
        backgroundColor: palette.surface,
        borderColor: palette.border,
        textStyle: { color: palette.ink },
        formatter: (p: any) => {
          const r = rows[p[0].dataIndex];
          return `<b>${modelLabels[r.model] ?? r.model}</b> · rank ${r.rank}<br/>MAE ${r.mae.toFixed(
            3
          )} ± ${r.mae_std.toFixed(3)}<br/>RMSE ${r.rmse.toFixed(3)}<br/>WAPE ${r.wape.toFixed(1)}%`;
        },
      },
      xAxis: {
        type: "category",
        data: rows.map((r) => modelLabels[r.model] ?? r.model),
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      yAxis: {
        type: "value",
        name: METRIC_LABEL[metric] + (metric === "wape" ? " (%)" : ""),
        nameTextStyle: { color: palette.faint },
        scale: metric !== "wape",
        splitLine: { lineStyle: { color: palette.grid } },
        axisLabel: { color: palette.muted },
      },
      series: [
        {
          type: "bar",
          data: rows.map((r, i) => ({
            value: r[metric],
            itemStyle: {
              color: i === 0 ? palette.accent : modelColors[r.model] ?? palette.faint,
              borderRadius: [4, 4, 0, 0],
            },
          })),
          barWidth: "52%",
        },
      ],
    };
  }, [rows, metric]);

  return (
    <Section
      id="ettm"
      eyebrow="External validity · 15-minute ETTm"
      title="Does the finding transfer to 15-minute data?"
      lead="The core comparison runs on hourly ETTh. Here the same linear-vs-Transformer question is re-asked on the 15-minute ETTm1 / ETTm2 datasets (horizon 96) as an external-validity check."
      tint
    >
      <div className="mb-8 flex flex-wrap gap-x-8 gap-y-4">
        <Segmented label="Dataset" value={dataset} options={["ETTm1", "ETTm2"]} onChange={setDataset} />
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
          <p className="mt-2 px-1 text-xs text-faint">
            Models ranked by {METRIC_LABEL[metric]} on {dataset} (lower is better; best highlighted).
          </p>
        </div>

        <div className="card flex flex-col p-6">
          <p className="eyebrow mb-4">Ranking</p>
          <ol className="space-y-2">
            {rows.map((r, i) => (
              <li key={r.model} className="flex items-baseline justify-between text-sm">
                <span className={i === 0 ? "font-medium text-ink" : "text-muted"}>
                  <span className="mr-2 font-mono text-faint">{i + 1}</span>
                  {modelLabels[r.model] ?? r.model}
                </span>
                <span className="font-mono text-ink">
                  {metric === "wape" ? r.wape.toFixed(1) + "%" : r[metric].toFixed(3)}
                </span>
              </li>
            ))}
          </ol>

          <div className="mt-5 rounded-xl bg-accent/[0.06] p-4">
            <p className="text-sm leading-relaxed text-ink">
              The linear family carries over: <b>{best ? modelLabels[best.model] ?? best.model : "—"}</b> is
              the best model on {dataset}, while the Transformer sits at rank {transformer?.rank ?? "—"}.
              The advantage transfers to minute-level forecasting — though on ETTm1 the top linear
              models are close.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
}
