// Small correlation helpers for the accuracy-vs-detection analysis.
// Pure functions, no dependencies.

export function pearson(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return NaN;
  const mx = mean(x);
  const my = mean(y);
  let cov = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    cov += dx * dy;
    sx += dx * dx;
    sy += dy * dy;
  }
  if (sx === 0 || sy === 0) return NaN;
  return cov / Math.sqrt(sx * sy);
}

// Spearman rank correlation = Pearson on average ranks (ties averaged).
export function spearman(x: number[], y: number[]): number {
  return pearson(rank(x), rank(y));
}

function mean(a: number[]): number {
  return a.reduce((s, v) => s + v, 0) / a.length;
}

function rank(a: number[]): number[] {
  const idx = a.map((v, i) => [v, i] as [number, number]).sort((p, q) => p[0] - q[0]);
  const ranks = new Array<number>(a.length);
  let i = 0;
  while (i < idx.length) {
    let j = i;
    while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
    // average rank (1-based) for the tie group [i..j]
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k][1]] = avg;
    i = j + 1;
  }
  return ranks;
}

export function fmtR(r: number): string {
  return Number.isNaN(r) ? "—" : (r >= 0 ? "+" : "") + r.toFixed(2);
}
