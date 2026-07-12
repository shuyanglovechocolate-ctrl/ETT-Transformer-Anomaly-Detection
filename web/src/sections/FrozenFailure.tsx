import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { FrozenData } from "../data/types";
import { palette } from "../theme";

export default function FrozenFailure() {
  const { data } = useJson<FrozenData>("frozen.json");
  const detectors = data?.detectors ?? [];
  const diag = data?.diagnosis;
  const residual = detectors.find((d) => d.name === "residual");
  const hybrid = detectors.find((d) => d.name === "hybrid_or");
  const flatness = detectors.find((d) => d.name === "flatness");

  const option: EChartsOption = useMemo(() => {
    return {
      grid: { left: 8, right: 20, top: 34, bottom: 8, containLabel: true },
      legend: { data: ["F1", "Event recall"], top: 2, textStyle: { color: palette.muted } },
      tooltip: {
        trigger: "axis",
        backgroundColor: palette.surface,
        borderColor: palette.border,
        textStyle: { color: palette.ink },
      },
      xAxis: {
        type: "category",
        data: detectors.map((d) => d.label),
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      yAxis: {
        type: "value",
        max: 1,
        splitLine: { lineStyle: { color: palette.grid } },
        axisLabel: { color: palette.muted },
      },
      series: [
        {
          name: "F1",
          type: "bar",
          data: detectors.map((d) => d.f1),
          itemStyle: { color: palette.faint, borderRadius: [3, 3, 0, 0] },
          barWidth: "30%",
        },
        {
          name: "Event recall",
          type: "bar",
          data: detectors.map((d, i) => ({
            value: d.event_recall,
            itemStyle: {
              color: i === 0 ? palette.danger : palette.accent,
              borderRadius: [3, 3, 0, 0],
            },
          })),
          barWidth: "30%",
        },
      ],
    };
  }, [detectors]);

  return (
    <Section
      id="frozen"
      title="The frozen blind spot — and a flatness fix"
      lead="The residual detector is strong on spikes and level shifts but collapses on frozen (stuck-sensor) segments. Rather than hide it, the study diagnoses why and adds a targeted fix."
    >
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          <p className="mb-3 px-1 text-sm text-muted">
            Detection on <b className="text-ink">frozen</b> anomalies — the residual detector barely
            registers events; a flatness diagnostic and the hybrid recover them.
          </p>
          <EChart option={option} height={340} />
        </div>

        <div className="card flex flex-col p-6">
          <p className="eyebrow mb-4">Why residuals fail</p>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl bg-surface-2 p-4">
              <p className="font-mono text-3xl font-semibold text-accent">
                {diag ? `${diag.flatness_ratio}×` : "—"}
              </p>
              <p className="mt-1 text-xs leading-snug text-muted">
                flatness signal, anomaly vs normal
              </p>
            </div>
            <div className="rounded-xl bg-surface-2 p-4">
              <p className="font-mono text-3xl font-semibold text-faint">
                {diag ? `${diag.residual_ratio}×` : "—"}
              </p>
              <p className="mt-1 text-xs leading-snug text-muted">
                residual signal, anomaly vs normal
              </p>
            </div>
          </div>

          <div className="mt-5 space-y-3 border-t border-border/70 pt-4">
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-muted">Event recall</span>
              <span className="font-mono text-sm text-ink">
                {residual && hybrid
                  ? `${Math.round(residual.event_recall * 100)}% → ${Math.round(
                      hybrid.event_recall * 100
                    )}%`
                  : "—"}
              </span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-muted">Best F1 on frozen</span>
              <span className="font-mono text-sm text-ink">
                {residual && flatness
                  ? `${residual.f1.toFixed(2)} → ${flatness.f1.toFixed(2)}`
                  : "—"}
              </span>
            </div>
          </div>

          <div className="mt-5 rounded-xl bg-accent/[0.06] p-4">
            <p className="text-sm leading-relaxed text-ink">
              A stuck sensor tracks its own flat value, so residuals stay small — yet the segment is
              ~{diag?.flatness_ratio ?? 24}× flatter than normal. A flatness diagnostic recovers what
              the residual score structurally cannot see.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
}
