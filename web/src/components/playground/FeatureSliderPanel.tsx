"use client";

import { memo } from "react";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { RotateCcw } from "lucide-react";
import type { FeatureMeta } from "@/lib/types";
import { cn } from "@/lib/utils";

interface FeatureSliderPanelProps {
  adjustableFeatures: string[];
  featureMeta: Record<string, FeatureMeta>;
  values: Record<string, number>;
  baseline: Record<string, number>;
  onChange: (feature: string, value: number) => void;
  onCommit: (feature: string, value: number) => void;
  onReset: () => void;
}

export const FeatureSliderPanel = memo(function FeatureSliderPanel({
  adjustableFeatures,
  featureMeta,
  values,
  baseline,
  onChange,
  onCommit,
  onReset,
}: FeatureSliderPanelProps) {
  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Design Parameters</h3>
        <Button variant="ghost" size="sm" onClick={onReset} className="h-7 text-xs">
          <RotateCcw className="mr-1 h-3 w-3" />
          Reset
        </Button>
      </div>

      {adjustableFeatures.map((feat, index) => {
        const meta = featureMeta[feat];
        if (!meta) return null;

        const currentVal = values[feat] ?? 0;
        const baseVal = baseline[feat] ?? currentVal;

        let min = meta.dataset_range?.min ?? 0;
        let max = meta.dataset_range?.max ?? 100;
        let step = meta.dataset_range?.step ?? 1;

        if (meta.bounds) {
          if (meta.bounds.scale === "x") {
            const baseRef = baseline[feat] ?? 1;
            if (meta.bounds.min != null) min = baseRef * meta.bounds.min;
            if (meta.bounds.max != null) max = baseRef * meta.bounds.max;
            if (meta.bounds.step != null)
              step = Math.abs(baseRef) * meta.bounds.step;
          } else {
            if (meta.bounds.min != null) min = meta.bounds.min;
            if (meta.bounds.max != null) max = meta.bounds.max;
            if (meta.bounds.step != null) step = meta.bounds.step;
          }
        }

        if (min > max) [min, max] = [max, min];
        const changed = Math.abs(currentVal - baseVal) > 0.001;

        return (
          <div key={feat} className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">
                {meta.label}
                {meta.unit && (
                  <span className="text-muted-foreground ml-1 text-xs">
                    ({meta.unit})
                  </span>
                )}
              </label>
              <span
                className={cn(
                  "text-sm font-mono tabular-nums",
                  changed && "text-primary font-semibold"
                )}
              >
                {currentVal.toFixed(step < 1 ? 2 : 1)}
              </span>
            </div>
            <Slider
              value={[currentVal]}
              min={min}
              max={max}
              step={step}
              onValueChange={([v]) => onChange(feat, v)}
              onValueCommit={([v]) => onCommit(feat, v)}
              className="w-full"
            />
            <div className="flex justify-between text-[10px] text-muted-foreground">
              <span>{min.toFixed(step < 1 ? 2 : 0)}</span>
              <span>{max.toFixed(step < 1 ? 2 : 0)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
});
