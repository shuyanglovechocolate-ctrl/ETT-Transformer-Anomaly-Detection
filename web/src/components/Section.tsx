import { motion } from "framer-motion";
import type { ReactNode } from "react";

// A vertical narrative section with a fade/rise-in on scroll.
export default function Section({
  id,
  eyebrow,
  title,
  lead,
  children,
  tint = false,
}: {
  id: string;
  eyebrow?: string;
  title: string;
  lead?: ReactNode;
  children?: ReactNode;
  tint?: boolean;
}) {
  return (
    <section
      id={id}
      className={`py-24 md:py-32 ${tint ? "bg-surface-2" : "bg-bg"}`}
    >
      <div className="mx-auto max-w-content px-6">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          {eyebrow && <p className="eyebrow mb-3">{eyebrow}</p>}
          <h2 className="max-w-2xl text-3xl font-semibold tracking-tight text-ink md:text-4xl">
            {title}
          </h2>
          {lead && <p className="mt-4 max-w-2xl text-lg leading-relaxed text-muted">{lead}</p>}
          <div className="mt-10">{children}</div>
        </motion.div>
      </div>
    </section>
  );
}
