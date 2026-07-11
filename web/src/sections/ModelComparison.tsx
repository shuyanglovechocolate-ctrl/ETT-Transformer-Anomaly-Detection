import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import Segmented from "../components/Segmented";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { ComparisonRow } from "../data/types";
import { palette, modelColors, modelLabels } from "../theme";

const MODEL_ORDER = ["naive", "linear", "nlinear", "dlinear", "lstm", "transformer"];

export default function ModelComparison() {
  const { data, loading } = useJson<ComparisonRow[]>("comparison.json");
  const [dataset, setDataset] = useState("ETTh1");
  const [inputType, setInputType] = useState<"multivariate" | "univariate">("multivariate");
  const [horizon, setHorizon] = useState(24);

  const rows = useMemo(() => {
    if (!data) return [];
    return data
      .filter(
        (r) =>
          r.dataset === dataset &&
          r.input_type === inputType &&
          r.horizon === horizon &&
          MODEL_ORDER.includes(r.model)
      )
      .sort((a, b) => MODEL_ORDER.indexOf(a.model) - MODEL_ORDER.indexOf(b.model));
  }, [data, dataset, inputType, horizon]);

  const best = rows.length ? Math.min(...rows.map((r) => r.mae)) : 0;

  const option: EChartsOption = useMemo(
    () => ({
      grid: { left: 8, right: 24, top: 20, bottom: 8, containLabel: true },
      tooltip: {
        trigger: "axis",
        backgroundColor: palette.surface,
        borderColor: palette.border,
        textStyle: { color: palette.ink },
        formatter: (p: any) => {
          const r = rows[p[0].dataIndex];
          return `<b>${modelLabels[r.model]}</b><br/>MAE ${r.mae.toFixed(3)} ± ${r.mae_std.toFixed(
            3
          )}<br/>RMSE ${r.rmse.toFixed(3)}<br/>WAPE ${r.wape.toFixed(1)}%`;
        },
      },
      xAxis: {
        type: "category",
        data: rows.map((r) => modelLabels[r.model]),
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      yAxis: {
        type: "value",
        name: "MAE",
        nameTextStyle: { color: palette.faint },
        splitLine: { lineStyle: { color: palette.grid } },
        axisLabel: { color: palette.muted },
      },
      // custom whisker series has a renderItem return ECharts types too strictly
      series: [
        {
          type: "bar",
          data: rows.map((r) => ({
            value: r.mae,
            itemStyle: {
              color: r.mae === best ? palette.accent : modelColors[r.model],
              borderRadius: [4, 4, 0, 0],
            },
          })),
          barWidth: "52%",
        },
        {
          // error bars (±std) via a custom whisker
          type: "custom",
          renderItem: (_params: any, api: any) => {
            const idx = api.value(0);
            const r = rows[idx];
            const x = api.coord([idx, r.mae])[0];
            const yTop = api.coord([idx, r.mae + r.mae_std])[1];
            const yBot = api.coord([idx, r.mae - r.mae_std])[1];
            const w = 6;
            const line = (y: number) => ({
              type: "line",
              shape: { x1: x - w, y1: y, x2: x + w, y2: y },
              style: { stroke: palette.ink, lineWidth: 1 },
            });
            return {
              type: "group",
              children: [
                { type: "line", shape: { x1: x, y1: yTop, x2: x, y2: yBot }, style: { stroke: palette.ink, lineWidth: 1 } },
                line(yTop),
                line(yBot),
              ],
            };
          },
          data: rows.map((_, i) => [i]),
          z: 10,
        } as any,
      ],
    }),
    [rows, best]
  );

  return (
    <Section
      id="forecasting"
      eyebrow="Module 3 · Forecasting"
      title="Do deeper models actually beat linear baselines?"
      lead="Mean absolute error on oil temperature, averaged over 3 seeds (whiskers = ±1 std). Switch dataset, input type and horizon — the linear family stays remarkably hard to beat."
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
      <div className="card p-5">
        {loading ? (
          <div className="flex h-[360px] items-center justify-center text-faint">Loading…</div>
        ) : rows.length ? (
          <EChart option={option} height={360} />
        ) : (
          <div className="flex h-[360px] items-center justify-center text-faint">No data for this configuration.</div>
        )}
      </div>
    </Section>
  );
}
