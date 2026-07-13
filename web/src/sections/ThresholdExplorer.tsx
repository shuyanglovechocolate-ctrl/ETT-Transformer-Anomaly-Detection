import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { AnomalyData } from "../data/types";
import { palette } from "../theme";

type Row = AnomalyData["by_threshold"][number];

const METHOD_LABEL: Record<string, string> = {
  percentile: "Percentile",
  mad: "MAD",
  iqr: "IQR",
  mean_std: "Mean + k·σ",
};

// argmin / argmax over a metric, returning the winning row.
const best = (rows: Row[], key: keyof Row, dir: "max" | "min" = "max") =>
  rows.reduce((b, r) => ((dir === "max" ? r[key] > b[key] : r[key] < b[key]) ? r : b), rows[0]);

export default function ThresholdExplorer() {
  const { data } = useJson<AnomalyData>("anomaly.json");
  // Sort by F1 descending so the strongest rule reads first.
  const rows = useMemo(
    () => [...(data?.by_threshold ?? [])].sort((a, b) => b.f1 - a.f1),
    [data]
  );

  const verdict = useMemo(() => {
    if (!rows.length) return null;
    return {
      f1: best(rows, "f1", "max"),
      recall: best(rows, "recall", "max"),
      precision: best(rows, "precision", "max"),
      fpr: best(rows, "fpr", "min"),
    };
  }, [rows]);

  const option: EChartsOption = useMemo(() => {
    const cats = rows.map((r) => METHOD_LABEL[r.method] ?? r.method);
    const mk = (name: string, key: keyof Row, color: string) => ({
      name,
      type: "bar" as const,
      data: rows.map((r) => r[key] as number),
      itemStyle: { color, borderRadius: [3, 3, 0, 0] },
    });
    return {
      grid: { left: 8, right: 20, top: 30, bottom: 8, containLabel: true },
      legend: { data: ["Precision", "Recall", "F1"], top: 0, textStyle: { color: palette.muted } },
      tooltip: {
        trigger: "axis",
        backgroundColor: palette.surface,
        borderColor: palette.border,
        textStyle: { color: palette.ink },
        formatter: (ps: any) => {
          const r = rows[ps[0].dataIndex];
          return `<b>${METHOD_LABEL[r.method] ?? r.method}</b><br/>
            Precision ${r.precision.toFixed(3)} ± ${r.precision_std.toFixed(2)}<br/>
            Recall ${r.recall.toFixed(3)} ± ${r.recall_std.toFixed(2)}<br/>
            F1 ${r.f1.toFixed(3)} ± ${r.f1_std.toFixed(2)}<br/>
            FPR ${r.fpr.toFixed(3)}`;
        },
      },
      xAxis: {
        type: "category",
        data: cats,
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      yAxis: {
        type: "value",
        max: 0.6,
        splitLine: { lineStyle: { color: palette.grid } },
        axisLabel: { color: palette.muted },
      },
      series: [
        mk("Precision", "precision", palette.faint),
        mk("Recall", "recall", palette.accent2),
        mk("F1", "f1", palette.accent),
      ],
    };
  }, [rows]);

  return (
    <Section
      id="threshold"
      eyebrow="Anomaly detection · thresholding"
      title="The threshold rule is a confound, not a footnote"
      lead="Turning a continuous residual score into a binary alarm needs a cutoff — and the rule you pick silently trades precision for recall. The same residual detector, scored four common ways over every anomaly type and seed, lands in very different places."
    >
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          {rows.length ? (
            <EChart option={option} height={360} />
          ) : (
            <div className="flex h-[360px] items-center justify-center text-faint">Loading…</div>
          )}
        </div>

        <div className="card flex flex-col p-6">
          <p className="eyebrow mb-4">No free lunch</p>
          {verdict && (
            <ul className="space-y-3 text-sm">
              <li className="flex items-baseline justify-between gap-3">
                <span className="text-muted">Best F1</span>
                <span className="text-ink">
                  <b>{METHOD_LABEL[verdict.f1.method]}</b>
                  <span className="ml-2 font-mono text-faint">{verdict.f1.f1.toFixed(3)}</span>
                </span>
              </li>
              <li className="flex items-baseline justify-between gap-3">
                <span className="text-muted">Max recall</span>
                <span className="text-ink">
                  <b>{METHOD_LABEL[verdict.recall.method]}</b>
                  <span className="ml-2 font-mono text-faint">{verdict.recall.recall.toFixed(3)}</span>
                </span>
              </li>
              <li className="flex items-baseline justify-between gap-3">
                <span className="text-muted">Lowest false-alarm rate</span>
                <span className="text-ink">
                  <b>{METHOD_LABEL[verdict.fpr.method]}</b>
                  <span className="ml-2 font-mono text-faint">{verdict.fpr.fpr.toFixed(3)}</span>
                </span>
              </li>
            </ul>
          )}
          <div className="mt-5 rounded-xl bg-accent/[0.06] p-4">
            <p className="text-sm leading-relaxed text-ink">
              No rule wins on every axis: the recall-maximizing choice raises the false-alarm rate, and the
              F1-best choice gives up recall to gain precision. Because the cutoff is a confound, the stress
              test above reports <b>threshold-free</b> PR-AUC / ROC-AUC — separating detector quality from the
              rule used to read it.
            </p>
          </div>
        </div>
      </div>
    </Section>
  );
}
