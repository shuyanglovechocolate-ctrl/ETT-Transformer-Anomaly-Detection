import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { AnomalyData } from "../data/types";
import { palette, anomalyColors } from "../theme";

const TYPE_LABEL: Record<string, string> = {
  spike: "Spike",
  level_shift: "Level shift",
  frozen: "Frozen",
};

export default function AnomalyDetection() {
  const { data } = useJson<AnomalyData>("anomaly.json");
  const byType = data?.by_type ?? [];

  const option: EChartsOption = useMemo(() => {
    const order = ["spike", "level_shift", "frozen"];
    const rows = [...byType].sort((a, b) => order.indexOf(a.anomaly_type) - order.indexOf(b.anomaly_type));
    return {
      grid: { left: 8, right: 24, top: 30, bottom: 8, containLabel: true },
      legend: { data: ["Precision", "Recall", "F1"], textStyle: { color: palette.muted }, top: 0 },
      tooltip: { trigger: "axis", backgroundColor: palette.surface, borderColor: palette.border, textStyle: { color: palette.ink } },
      xAxis: {
        type: "category",
        data: rows.map((r) => TYPE_LABEL[r.anomaly_type]),
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      yAxis: { type: "value", max: 1, splitLine: { lineStyle: { color: palette.grid } }, axisLabel: { color: palette.muted } },
      series: [
        { name: "Precision", type: "bar", data: rows.map((r) => r.mean_precision), itemStyle: { color: palette.faint, borderRadius: [3, 3, 0, 0] } },
        { name: "Recall", type: "bar", data: rows.map((r) => r.mean_recall), itemStyle: { color: palette.accent2, borderRadius: [3, 3, 0, 0] } },
        {
          name: "F1",
          type: "bar",
          data: rows.map((r) => ({ value: r.best_mean_f1, itemStyle: { color: anomalyColors[r.anomaly_type], borderRadius: [3, 3, 0, 0] } })),
        },
      ],
    };
  }, [byType]);

  const frozen = byType.find((r) => r.anomaly_type === "frozen");

  return (
    <Section
      id="anomaly"
      eyebrow="Module 4 · Residual anomaly detection"
      title="The residual signal works — until it doesn't"
      lead="Reusing forecast residuals as an anomaly score cleanly catches spikes and level shifts (high recall), but collapses on frozen segments, where a flat signal produces small residuals. Reporting where the method fails is part of the contribution."
    >
      <div className="grid gap-6 md:grid-cols-3">
        <div className="card p-5 md:col-span-2">
          <EChart option={option} height={340} />
        </div>
        <div className="card flex flex-col justify-center p-6">
          <p className="eyebrow">Failure mode</p>
          <p className="stat-num mt-4 text-3xl" style={{ color: anomalyColors.frozen }}>
            {frozen ? frozen.best_mean_f1.toFixed(2) : "—"}
          </p>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            best mean F1 on <b className="text-ink">frozen</b> anomalies — versus{" "}
            {byType.find((r) => r.anomaly_type === "spike")?.best_mean_f1.toFixed(2)} on spikes. A flat
            output barely perturbs the residual, so a diagnostic flatness detector is needed instead.
          </p>
        </div>
      </div>
    </Section>
  );
}
