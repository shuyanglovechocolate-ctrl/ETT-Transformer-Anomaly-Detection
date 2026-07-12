import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import Segmented from "../components/Segmented";
import EChart from "../components/EChart";
import { useJson, dataUrl } from "../data/useJson";
import type { PredIndexEntry, PredSeries } from "../data/types";
import { palette, modelLabels } from "../theme";

export default function ForecastViz() {
  const { data: index } = useJson<PredIndexEntry[]>("predictions/index.json");
  const [dataset, setDataset] = useState("ETTh1");
  const [model, setModel] = useState("transformer");
  const [series, setSeries] = useState<PredSeries | null>(null);

  const models = useMemo(
    () => (index ? [...new Set(index.filter((e) => e.dataset === dataset).map((e) => e.model))] : []),
    [index, dataset]
  );

  useEffect(() => {
    let alive = true;
    const key = `${dataset}_${model}`;
    fetch(dataUrl(`predictions/${key}.json`))
      .then((r) => (r.ok ? r.json() : null))
      .then((d: PredSeries | null) => alive && setSeries(d))
      .catch(() => alive && setSeries(null));
    return () => {
      alive = false;
    };
  }, [dataset, model]);

  const option: EChartsOption = useMemo(() => {
    if (!series) return {};
    return {
      grid: { left: 8, right: 24, top: 24, bottom: 64, containLabel: true },
      legend: {
        data: ["Actual", "Predicted"],
        textStyle: { color: palette.muted },
        top: 0,
        right: 0,
      },
      tooltip: {
        trigger: "axis",
        backgroundColor: palette.surface,
        borderColor: palette.border,
        textStyle: { color: palette.ink },
      },
      xAxis: {
        type: "category",
        data: series.dates,
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted, formatter: (v: string) => v.slice(5, 10) },
      },
      yAxis: {
        type: "value",
        name: "Oil temp",
        nameTextStyle: { color: palette.faint },
        scale: true,
        splitLine: { lineStyle: { color: palette.grid } },
        axisLabel: { color: palette.muted },
      },
      dataZoom: [
        { type: "inside", start: 0, end: 30 },
        {
          type: "slider",
          start: 0,
          end: 30,
          height: 22,
          bottom: 12,
          borderColor: palette.border,
          fillerColor: palette.zoomFill,
          handleStyle: { color: palette.accent },
          textStyle: { color: palette.faint },
          dataBackground: { lineStyle: { color: palette.border }, areaStyle: { color: palette.grid } },
        },
      ],
      series: [
        {
          name: "Actual",
          type: "line",
          data: series.y_true,
          showSymbol: false,
          lineStyle: { width: 1.5, color: palette.muted },
        },
        {
          name: "Predicted",
          type: "line",
          data: series.y_pred,
          showSymbol: false,
          lineStyle: { width: 1.8, color: palette.accent },
        },
      ],
    };
  }, [series]);

  return (
    <Section
      id="forecast-viz"
      title="See a forecaster track the real signal"
      lead="First-step predictions vs. ground-truth oil temperature on the held-out test set. Drag the slider or scroll to zoom into any window."
    >
      <div className="mb-8 flex flex-wrap gap-x-8 gap-y-4">
        <Segmented label="Dataset" value={dataset} options={["ETTh1", "ETTh2"]} onChange={setDataset} />
        <Segmented
          label="Model"
          value={model}
          options={models.length ? models : [model]}
          onChange={setModel}
          format={(v) => modelLabels[v] ?? v}
        />
      </div>
      <div className="card p-5">
        {series ? (
          <EChart option={option} height={400} />
        ) : (
          <div className="flex h-[400px] items-center justify-center text-faint">Loading…</div>
        )}
      </div>
    </Section>
  );
}
