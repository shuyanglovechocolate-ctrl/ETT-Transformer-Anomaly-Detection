import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import Section from "../components/Section";
import EChart from "../components/EChart";
import { useJson } from "../data/useJson";
import type { LatencyData } from "../data/types";
import { palette, modelColors, modelLabels } from "../theme";

export default function LatencyCost() {
  const { data } = useJson<LatencyData>("latency.json");
  const records = data?.records ?? [];

  // Sort fastest → slowest; a horizontal bar makes the order-of-magnitude gap read at a glance.
  const rows = useMemo(
    () => [...records].sort((a, b) => a.latency_ms_per_1k - b.latency_ms_per_1k),
    [records]
  );

  // Headline: how much slower is the slowest model than the fastest *trained* one
  // (skip naive — it is a no-op baseline with zero parameters).
  const headline = useMemo(() => {
    if (rows.length < 2) return null;
    const trained = rows.filter((r) => r.params > 0);
    const fastest = trained[0];
    const slowest = rows[rows.length - 1];
    if (!fastest || !slowest) return null;
    return { fastest, slowest, ratio: slowest.latency_ms_per_1k / fastest.latency_ms_per_1k };
  }, [rows]);

  const option: EChartsOption = useMemo(() => {
    return {
      grid: { left: 8, right: 60, top: 10, bottom: 24, containLabel: true },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: palette.surface,
        borderColor: palette.border,
        textStyle: { color: palette.ink },
        formatter: (ps: any) => {
          const r = rows[ps[0].dataIndex];
          return `<b>${modelLabels[r.model] ?? r.model}</b><br/>
            ${r.latency_ms_per_1k.toFixed(1)} ± ${r.latency_std.toFixed(1)} ms / 1k windows<br/>
            ${r.latency_ms_per_batch.toFixed(2)} ms / batch<br/>
            ${r.params.toLocaleString()} params`;
        },
      },
      xAxis: {
        type: "value",
        name: "ms / 1k windows",
        nameLocation: "middle",
        nameGap: 32,
        nameTextStyle: { color: palette.faint, fontSize: 12 },
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
        splitLine: { lineStyle: { color: palette.grid } },
      },
      yAxis: {
        type: "category",
        inverse: true,
        data: rows.map((r) => modelLabels[r.model] ?? r.model),
        axisLine: { lineStyle: { color: palette.border } },
        axisLabel: { color: palette.muted },
      },
      series: [
        {
          type: "bar",
          data: rows.map((r) => ({
            value: r.latency_ms_per_1k,
            itemStyle: { color: modelColors[r.model] ?? palette.faint, borderRadius: [0, 3, 3, 0] },
          })),
          barWidth: "58%",
          label: {
            show: true,
            position: "right",
            color: palette.muted,
            fontSize: 11,
            formatter: (p: any) => `${p.value.toFixed(0)} ms`,
          },
        },
      ],
    };
  }, [rows]);

  return (
    <Section
      id="latency"
      eyebrow="Practical deployment · inference cost"
      title="The accurate models are also the cheap ones"
      lead="Accuracy is only half the deployment story — runtime cost is the other half. Timing every trained model on the same hardware shows the gap is not subtle: the recurrent and attention models are an order of magnitude slower than the linear family they never manage to beat."
      tint
    >
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          {rows.length ? (
            <EChart option={option} height={340} />
          ) : (
            <div className="flex h-[340px] items-center justify-center text-faint">Loading…</div>
          )}
        </div>

        <div className="card flex flex-col justify-center p-6">
          {headline && (
            <>
              <p className="eyebrow">Speed penalty</p>
              <p className="stat-num mt-3 text-4xl" style={{ color: palette.danger }}>
                {headline.ratio.toFixed(0)}×
              </p>
              <p className="mt-2 text-sm leading-relaxed text-muted">
                <b className="text-ink">{modelLabels[headline.slowest.model] ?? headline.slowest.model}</b> is{" "}
                {headline.ratio.toFixed(0)}× slower than{" "}
                <b className="text-ink">{modelLabels[headline.fastest.model] ?? headline.fastest.model}</b> per
                1k windows — with no accuracy gain to justify it.
              </p>
              <div className="mt-5 rounded-xl bg-accent/[0.06] p-4">
                <p className="text-sm leading-relaxed text-ink">
                  Put beside the significance result — where NLinear beats the Transformer with a real margin —
                  the deployment verdict is unambiguous: the linear family is both more accurate and roughly an
                  order of magnitude cheaper to serve.
                </p>
              </div>
            </>
          )}
          {data && (
            <p className="mt-4 text-xs leading-relaxed text-faint">
              {data.dataset} · horizon {data.horizon} · {data.input_type} · {data.device} ·{" "}
              {data.num_repeats}× repeats
            </p>
          )}
        </div>
      </div>
    </Section>
  );
}
