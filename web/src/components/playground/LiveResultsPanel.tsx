"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricCard } from "@/components/shared/MetricCard";
import { VerdictBadge } from "@/components/shared/VerdictBadge";
import { ProbabilityBars } from "@/components/charts/ProbabilityBars";
import type { PlaygroundResponse } from "@/lib/types";
import { Loader2 } from "lucide-react";

interface LiveResultsPanelProps {
  result: PlaygroundResponse | null;
  isLoading: boolean;
}

export function LiveResultsPanel({ result, isLoading }: LiveResultsPanelProps) {
  if (!result) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
        {isLoading ? (
          <Loader2 className="h-5 w-5 animate-spin" />
        ) : (
          "Adjust sliders to see predictions"
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h3 className="text-sm font-semibold">Live Prediction</h3>
        <VerdictBadge gap={result.gap_minutes} />
        {isLoading && <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <MetricCard
          label="Predicted Mode"
          value={result.predicted_mode}

        />
        <MetricCard
          label="P(No Failure)"
          value={(result.probabilities["No Failure"] ?? 0) * 100}
          suffix="%"
          decimals={2}
          delta={
            result.deltas
              ? result.deltas.prob_no_failure_delta * 100
              : undefined
          }
          deltaSuffix=" pp"
        />
        <MetricCard
          label="Fire Resistance"
          value={result.frt_minutes}
          suffix="min"
          delta={result.deltas?.frt_delta}
          deltaSuffix=" min"
        />
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Class Probabilities</CardTitle>
        </CardHeader>
        <CardContent>
          <ProbabilityBars
            probabilities={result.probabilities}
            predictedMode={result.predicted_mode}
          />
        </CardContent>
      </Card>

      {result.deltas?.mode_changed && (
        <div className="rounded-lg bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 p-3">
          <p className="text-sm text-amber-800 dark:text-amber-300">
            Mode changed from{" "}
            <strong>{result.deltas.previous_mode}</strong> to{" "}
            <strong>{result.predicted_mode}</strong>
          </p>
        </div>
      )}
    </div>
  );
}
