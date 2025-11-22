"use client";

import { MODE_COLORS } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface ProbabilityBarsProps {
  probabilities: Record<string, number>;
  predictedMode: string;
  className?: string;
}

export function ProbabilityBars({
  probabilities,
  predictedMode,
  className,
}: ProbabilityBarsProps) {
  const entries = Object.entries(probabilities).sort(
    ([, a], [, b]) => b - a
  );

  return (
    <div className={cn("space-y-3", className)}>
      {entries.map(([mode, prob]) => {
        const pct = prob * 100;
        const barWidth = Math.max(pct, 3);
        const isPredicted = mode === predictedMode;
        const color = MODE_COLORS[mode] || "#94a3b8";

        return (
          <div key={mode}>
            <div className="flex items-center justify-between mb-1">
              <span
                className={cn(
                  "text-sm",
                  isPredicted ? "font-semibold" : "text-muted-foreground"
                )}
              >
                {mode}
                {isPredicted && (
                  <span className="ml-1.5 text-xs text-primary font-normal">
                    predicted
                  </span>
                )}
              </span>
              <span className="text-sm font-mono tabular-nums">
                {pct.toFixed(1)}%
              </span>
            </div>
            <div className="h-3 w-full rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full transition-[width] duration-500 ease-out"
                style={{ backgroundColor: color, width: `${barWidth}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
