import { useState } from "react";

const REPO_URL =
  "https://github.com/shuyanglovechocolate-ctrl/ETT-Transformer-Anomaly-Detection";

const BIBTEX = `@mastersthesis{gao2026ett,
  author = {Shuyang Gao},
  title  = {Residual-based Transformer Forecasting for Oil Temperature Anomaly Detection},
  school = {University of Warwick},
  type   = {MSc dissertation},
  year   = {2026},
  url    = {${REPO_URL}}
}`;

export default function Citation() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    let ok = false;
    try {
      await navigator.clipboard.writeText(BIBTEX);
      ok = true;
    } catch {
      // fallback for restricted contexts (iframes, no-gesture, older browsers)
      try {
        const ta = document.createElement("textarea");
        ta.value = BIBTEX;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        ok = document.execCommand("copy");
        document.body.removeChild(ta);
      } catch {
        ok = false;
      }
    }
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <section id="citation" className="border-t border-border/60 bg-surface-2 py-20 md:py-24">
      <div className="mx-auto max-w-content px-6">
        <h2 className="text-2xl font-semibold tracking-tight text-ink md:text-3xl">Citation</h2>
        <p className="mt-3 max-w-2xl text-muted">If you use this work, please cite it as:</p>

        {/* human-readable reference */}
        <p className="mt-6 max-w-3xl leading-relaxed text-ink">
          Gao, S. (2026). <em>Residual-based Transformer Forecasting for Oil Temperature Anomaly
          Detection</em> [MSc dissertation, University of Warwick]. GitHub.{" "}
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="text-accent underline-offset-2 hover:underline">
            {REPO_URL.replace("https://", "")}
          </a>
        </p>

        {/* BibTeX */}
        <div className="mt-8 overflow-hidden rounded-xl border border-border bg-surface">
          <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
            <span className="font-mono text-xs uppercase tracking-wider text-faint">BibTeX</span>
            <button
              type="button"
              onClick={copy}
              className="rounded-md border border-border px-3 py-1.5 text-sm text-muted transition-colors hover:border-accent/50 hover:text-ink"
            >
              {copied ? "Copied ✓" : "Copy BibTeX"}
            </button>
          </div>
          <pre className="overflow-x-auto px-4 py-4 font-mono text-xs leading-relaxed text-ink">
            {BIBTEX}
          </pre>
        </div>

        <dl className="mt-8 grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2">
          <div className="flex gap-2">
            <dt className="text-faint">Repository</dt>
            <dd>
              <a href={REPO_URL} target="_blank" rel="noreferrer" className="text-accent underline-offset-2 hover:underline">
                GitHub ↗
              </a>
            </dd>
          </div>
          <div className="flex gap-2">
            <dt className="text-faint">DOI</dt>
            <dd className="text-muted">to be assigned (Zenodo, on release)</dd>
          </div>
        </dl>
      </div>
    </section>
  );
}
