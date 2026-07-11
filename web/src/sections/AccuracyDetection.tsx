import { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import EChart from "../components/EChart";
import Section from "../components/Section";
import { useJson } from "../data/useJson";
import type { AnomalyData } from "../data/types";
import { anomalyColors, modelColors, modelLabels, palette } from "../theme";

const ALL = "all";

function pearson(xs: number[], ys: number[]) {
