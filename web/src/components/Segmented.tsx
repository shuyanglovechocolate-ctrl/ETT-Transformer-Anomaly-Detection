// Minimal segmented control for filters. Controlled component.
// On narrow screens the label stacks above and the pill group scrolls within
// itself (never widening the page); touch targets are ~44px tall.
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
    <div className="flex max-w-full flex-col gap-1.5 sm:flex-row sm:items-center sm:gap-3">
      <span className="font-mono text-xs uppercase tracking-wider text-faint">{label}</span>
      <div className="flex max-w-full flex-nowrap overflow-x-auto rounded-lg border border-border bg-surface p-1">
        {options.map((opt) => {
          const active = opt === value;
          return (
            <button
              key={String(opt)}
              onClick={() => onChange(opt)}
              aria-pressed={active}
              className={`shrink-0 whitespace-nowrap rounded-md px-3.5 py-2.5 text-sm transition-colors ${
                active ? "bg-accent/15 text-accent" : "text-muted hover:text-ink"
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
