// Minimal segmented control for filters. Controlled component.
export default function Segmented<T extends string | number>({
  label,
  value,
  options,
  onChange,
  format,
}: {
  label: string;
  value: T;
  options: T[];
  onChange: (v: T) => void;
  format?: (v: T) => string;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-xs uppercase tracking-wider text-faint">{label}</span>
      <div className="inline-flex rounded-lg border border-border bg-surface p-1">
        {options.map((opt) => {
          const active = opt === value;
          return (
            <button
              key={String(opt)}
              onClick={() => onChange(opt)}
              className={`rounded-md px-3 py-1.5 text-sm transition-colors ${
                active
                  ? "bg-accent/15 text-accent"
                  : "text-muted hover:text-ink"
              }`}
            >
              {format ? format(opt) : String(opt)}
            </button>
          );
        })}
      </div>
    </div>
  );
}
