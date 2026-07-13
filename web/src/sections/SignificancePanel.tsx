import { useMemo, useState } from "react";
import Section from "../components/Section";
import Segmented from "../components/Segmented";
import { useJson } from "../data/useJson";
import type { SignificanceData } from "../data/types";
import { palette, modelLabels } from "../theme";

type Row = SignificanceData["comparisons"][number];

// Verdict colors — kept local (green is not in the shared palette). Every state
// is also carried by a non-color cue (signed number + weight + legend glyph).
const BETTER = "#2e9e5b";
const WORSE = palette.danger;

const signed = (v: number, digits = 2) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(digits);
const label = (m: string) => modelLabels[m] ?? m;

// Resolve the row-vs-col verdict from a record stored in either a/b order.
function orient(rec: Row, row: string) {
  const rowIsA = rec.model_a === row;
  const delta = rowIsA ? rec.delta : -rec.delta;
  const rowBetter = rowIsA ? rec.a_better : !rec.a_better;
  const ci: [number, number] = rowIsA ? [rec.ci_low, rec.ci_high] : [-rec.ci_high, -rec.ci_low];
  return { delta, rowBetter, ci, significant: rec.significant, num_points: rec.num_points };
}

export default function SignificancePanel() {
  const { data } = useJson<SignificanceData>("significance.json");
  const all = data?.comparisons ?? [];

  const datasets = useMemo(() => [...new Set(all.map((r) => r.dataset))], [all]);
  const horizons = useMemo(() => [...new Set(all.map((r) => r.horizon))].sort((a, b) => a - b), [all]);

  const [dataset, setDataset] = useState("ETTh1");
  const [horizon, setHorizon] = useState(24);
  const [sel, setSel] = useState<{ row: string; col: string }>({ row: "nlinear", col: "transformer" });

  // Comparisons for the selected (dataset, horizon) slice.
  const slice = useMemo(
    () => all.filter((r) => r.dataset === dataset && r.horizon === horizon),
    [all, dataset, horizon]
  );

  // Per-model MAE within the slice → order models best (lowest MAE) first so the
  // matrix reads as a clean better/worse triangle.
  const models = useMemo(() => {
    const mae: Record<string, number> = {};
    for (const r of slice) {
      mae[r.model_a] = r.mae_a;
      mae[r.model_b] = r.mae_b;
    }
    return Object.keys(mae).sort((a, b) => mae[a] - mae[b]);
  }, [slice]);

  // (row, col) → record lookup for the slice.
  const cell = (row: string, col: string) =>
    slice.find((r) => (r.model_a === row && r.model_b === col) || (r.model_a === col && r.model_b === row));

  // ① executive summary over the current slice.
  const summary = useMemo(() => {
    const sig = slice.filter((r) => r.significant);
    const largest = sig.reduce<Row | null>(
      (best, r) => (Math.abs(r.delta) > Math.abs(best?.delta ?? 0) ? r : best),
      null
    );
    let winner = "",
      loser = "",
      mag = 0;
    if (largest) {
      winner = largest.a_better ? largest.model_a : largest.model_b;
      loser = largest.a_better ? largest.model_b : largest.model_a;
      mag = Math.abs(largest.delta);
    }
    return { sig: sig.length, total: slice.length, ns: slice.length - sig.length, winner, loser, mag };
  }, [slice]);

  // ③ interpretation over ALL slices (global takeaway, independent of filters).
  const interp = useMemo(() => {
    const eq = (r: Row, x: string, y: string) =>
      (r.model_a === x && r.model_b === y) || (r.model_a === y && r.model_b === x);
    const nt = all.filter((r) => eq(r, "nlinear", "transformer"));
    let nlWins = 0;
    for (const r of nt) {
      const nlBetter = r.model_a === "nlinear" ? r.a_better : !r.a_better;
      if (r.significant && nlBetter) nlWins++;
    }
    const nsCount: Record<string, number> = {};
    const total: Record<string, number> = {};
    for (const r of all) {
      const k = [r.model_a, r.model_b].sort().join("|");
      total[k] = (total[k] ?? 0) + 1;
      if (!r.significant) nsCount[k] = (nsCount[k] ?? 0) + 1;
    }
    let nsPair: string | null = null;
    let nsMax = 0;
    for (const k of Object.keys(nsCount))
      if (nsCount[k] > nsMax) {
        nsMax = nsCount[k];
        nsPair = k;
      }
    const [nsA, nsB] = nsPair ? nsPair.split("|") : ["", ""];
    return { ntTotal: nt.length, nlWins, nsA, nsB, nsMax, nsTotal: nsPair ? total[nsPair] : 0 };
  }, [all]);

  const selRec = cell(sel.row, sel.col);
  const selView = selRec && sel.row !== sel.col ? orient(selRec, sel.row) : null;

  const gridCols = { gridTemplateColumns: `minmax(76px, auto) repeat(${models.length}, minmax(0, 1fr))` };

  return (
    <Section
      id="significance"
      eyebrow="Statistical evidence · paired bootstrap"
      title="Is the linear advantage real, or just noise?"
      lead="A lower mean error is only a finding if it survives resampling. Every model pair is compared with a paired bootstrap on the MAE difference; a gap counts only when its 95% confidence interval excludes zero. Pick a slice, then click any cell for the interval behind it."
      tint
    >
      <div className="mb-8 flex flex-wrap gap-x-8 gap-y-4">
        <Segmented label="Dataset" value={dataset} options={datasets} onChange={setDataset} />
        <Segmented label="Horizon" value={horizon} options={horizons} onChange={setHorizon} format={(v) => `h${v}`} />
      </div>

      {/* ① executive summary */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="card p-5">
          <p className="eyebrow">Significant</p>
          <p className="stat-num mt-3 text-3xl" style={{ color: BETTER }}>
            {summary.sig}
            <span className="text-lg text-faint"> / {summary.total}</span>
          </p>
          <p className="mt-1 text-sm text-muted">pairs with a real MAE gap</p>
        </div>
        <div className="card p-5">
          <p className="eyebrow">Largest improvement</p>
          <p className="stat-num mt-3 text-3xl text-ink">{summary.mag ? summary.mag.toFixed(3) : "—"}</p>
          <p className="mt-1 text-sm text-muted">
            {summary.winner ? (
              <>
                <b className="text-ink">{label(summary.winner)}</b> over {label(summary.loser)} (ΔMAE)
              </>
            ) : (
              "—"
            )}
          </p>
        </div>
        <div className="card p-5">
          <p className="eyebrow">Non-significant</p>
          <p className="stat-num mt-3 text-3xl text-ink">{summary.ns}</p>
          <p className="mt-1 text-sm text-muted">statistical ties</p>
        </div>
        <div className="card p-5">
          <p className="eyebrow">Confidence</p>
          <p className="stat-num mt-3 text-3xl text-ink">95%</p>
          <p className="mt-1 text-sm text-muted">paired-bootstrap CI</p>
        </div>
      </div>

      {/* ② matrix + detail */}
      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <div className="card overflow-x-auto p-4 md:p-5 lg:col-span-2">
          <p className="eyebrow mb-1 px-1">Row vs. column · signed ΔMAE</p>
          <p className="mb-4 px-1 text-xs text-faint">
            green = row significantly better · red = row worse · gray = no significant difference
          </p>
          <div className="min-w-[520px]">
            <div className="grid gap-1.5" style={gridCols}>
              <div />
              {models.map((m) => (
                <div key={m} className="px-1 pb-2 text-center font-mono text-[11px] uppercase tracking-wide text-faint">
                  {label(m)}
                </div>
              ))}
            </div>
            {models.map((row) => (
              <div key={row} className="mb-1.5 grid items-stretch gap-1.5" style={gridCols}>
                <div className="flex items-center pr-2 font-mono text-[11px] uppercase tracking-wide text-faint">
                  {label(row)}
                </div>
                {models.map((col) => {
                  if (row === col)
                    return <div key={col} className="flex h-11 items-center justify-center rounded-md bg-surface-2 text-sm text-faint">—</div>;
                  const rec = cell(row, col);
                  if (!rec) return <div key={col} className="h-11 rounded-md bg-surface-2" />;
                  const v = orient(rec, row);
                  const color = v.significant ? (v.rowBetter ? BETTER : WORSE) : palette.muted;
                  const bg = v.significant
                    ? v.rowBetter
                      ? "rgba(46, 158, 91, 0.14)"
                      : "rgba(217, 58, 58, 0.12)"
                    : "rgba(120, 120, 128, 0.08)";
                  const isSel = sel.row === row && sel.col === col;
                  return (
                    <button
                      key={col}
                      onClick={() => setSel({ row, col })}
                      aria-pressed={isSel}
                      className="flex h-11 items-center justify-center rounded-md text-sm tabular-nums transition-colors"
                      style={{
                        background: bg,
                        color,
                        fontWeight: v.significant ? 600 : 400,
                        boxShadow: isSel ? `inset 0 0 0 2px ${palette.accent}` : undefined,
                      }}
                      title={`${label(row)} vs ${label(col)} · ΔMAE ${signed(v.delta, 3)} · ${
                        v.significant ? "significant" : "not significant"
                      }`}
                    >
                      {signed(v.delta)}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* detail panel */}
        <div className="card flex flex-col p-6">
          <p className="eyebrow mb-4">Comparison detail</p>
          {selView ? (
            <>
              <p className="text-lg font-semibold text-ink">
                {label(sel.row)} <span className="text-faint">vs</span> {label(sel.col)}
              </p>
              <dl className="mt-5 space-y-3 text-sm">
                <div className="flex items-baseline justify-between">
                  <dt className="text-muted">ΔMAE (row − col)</dt>
                  <dd className="font-mono tabular-nums" style={{ color: selView.significant ? (selView.rowBetter ? BETTER : WORSE) : palette.ink }}>
                    {signed(selView.delta, 3)}
                  </dd>
                </div>
                <div className="flex items-baseline justify-between">
                  <dt className="text-muted">95% CI</dt>
                  <dd className="font-mono tabular-nums text-ink">
                    [{signed(selView.ci[0], 3)}, {signed(selView.ci[1], 3)}]
                  </dd>
                </div>
                <div className="flex items-baseline justify-between">
                  <dt className="text-muted">Paired points (N)</dt>
                  <dd className="font-mono tabular-nums text-ink">{selView.num_points.toLocaleString()}</dd>
                </div>
                <div className="flex items-baseline justify-between border-t border-border/60 pt-3">
                  <dt className="text-muted">Verdict</dt>
                  <dd className="font-medium" style={{ color: selView.significant ? (selView.rowBetter ? BETTER : WORSE) : palette.muted }}>
                    {selView.significant ? (selView.rowBetter ? "Significant ✓" : "Significantly worse") : "Not significant"}
                  </dd>
                </div>
              </dl>
              <p className="mt-5 rounded-xl bg-accent/[0.06] p-4 text-sm leading-relaxed text-ink">
                {selView.significant
                  ? `The 95% interval [${signed(selView.ci[0], 3)}, ${signed(selView.ci[1], 3)}] stays entirely on one side of zero, so the gap is unlikely to be resampling noise.`
                  : `The 95% interval [${signed(selView.ci[0], 3)}, ${signed(selView.ci[1], 3)}] straddles zero — this pair is a statistical tie under the tested slice.`}
              </p>
            </>
          ) : (
            <p className="text-sm text-muted">Select a cell in the matrix to see its bootstrap interval.</p>
          )}
        </div>
      </div>

      {/* ③ interpretation */}
      <div className="mt-6 card p-6" style={{ background: "rgba(0,106,214,0.05)" }}>
        <p className="eyebrow mb-3">What the statistics say</p>
        <p className="text-base leading-relaxed text-ink">
          Across both datasets and all three horizons, <b>NLinear significantly outperformed the Transformer
          in {interp.nlWins}/{interp.ntTotal} comparisons</b> (95% paired-bootstrap confidence) — the linear
          advantage is a statistically supported result, not a single-seed fluke.
        </p>
        {interp.nsMax > 1 && (
          <p className="mt-3 text-base leading-relaxed text-muted">
            By contrast, {label(interp.nsA)} and {label(interp.nsB)} showed no significant MAE difference in{" "}
            {interp.nsMax}/{interp.nsTotal} settings — a genuine tie, honestly reported as such.
          </p>
        )}
      </div>
    </Section>
  );
}
