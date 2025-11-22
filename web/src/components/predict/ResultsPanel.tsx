"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { MetricCard } from "@/components/shared/MetricCard";
import { VerdictBadge } from "@/components/shared/VerdictBadge";
import { ModeBadge } from "@/components/shared/ModeBadge";
import { ProbabilityBars } from "@/components/charts/ProbabilityBars";
import { ActionPlanTable } from "./ActionPlanTable";
import type { AnalyzeResponse } from "@/lib/types";

interface ResultsPanelProps {
  result: AnalyzeResponse;
}

export function ResultsPanel({ result }: ResultsPanelProps) {
  const { prediction, scenario, recommendations, combination, notes } = result;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-3 duration-400">
      <div className="flex items-center gap-3">
        <h2 className="text-lg font-semibold">Analysis Results</h2>
        <VerdictBadge gap={prediction.gap_minutes} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <MetricCard
          label="Predicted Mode"
          value={prediction.predicted_mode}

        />
        <MetricCard
          label="P(No Failure)"
          value={
            (prediction.probabilities["No Failure"] ?? 0) * 100
          }
          suffix="%"
          decimals={2}
        />
        <MetricCard
          label="Fire Resistance"
          value={prediction.frt_minutes}
          suffix="min"
          delta={prediction.gap_minutes}
          deltaSuffix=" min vs req."
        />
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Class Probabilities</CardTitle>
        </CardHeader>
        <CardContent>
          <ProbabilityBars
            probabilities={prediction.probabilities}
            predictedMode={prediction.predicted_mode}
          />
        </CardContent>
      </Card>

      <Separator />

      <ActionPlanTable
        recommendations={recommendations}
        combination={combination}
        notes={notes}
      />
    </div>
  );
}
