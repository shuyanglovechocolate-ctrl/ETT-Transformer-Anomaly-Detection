import { useEffect, useRef } from "react";
import * as echarts from "echarts";

// Thin React wrapper around Apache ECharts. Re-renders on option change and
// resizes with its container. Kept dependency-light (no echarts-for-react).
export default function EChart({
  option,
  height = 360,
  className = "",
}: {
  option: echarts.EChartsOption;
  height?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    chart.current = echarts.init(ref.current, undefined, { renderer: "canvas" });
    const onResize = () => chart.current?.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption(option, true);
  }, [option]);

  return <div ref={ref} style={{ height }} className={className} />;
}
