"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ModeBadge } from "@/components/shared/ModeBadge";
import type { RecommendationResponse, CombinationResponse } from "@/lib/types";
import { ArrowRight } from "lucide-react";

interface ActionPlanTableProps {
  recommendations: RecommendationResponse[];
  combination?: CombinationResponse | null;
  notes: string[];
}

export function ActionPlanTable({
  recommendations,
  combination,
  notes,
}: ActionPlanTableProps) {
  if (recommendations.length === 0 && !combination) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-sm text-green-600 dark:text-green-400">
            {notes[0] || "Beam already satisfies the scenario."}
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Recommendations</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {recommendations.map((rec, i) => (
            <div
              key={i}
              className="rounded-lg border p-3 space-y-2"
            >
              <div className="flex items-start justify-between gap-2">
                <p className="text-sm font-medium leading-snug">
                  {rec.action}
                </p>
                <ModeBadge mode={rec.new_mode} className="text-xs py-0.5 px-2 shrink-0" />
              </div>

              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="rounded-md bg-muted/50 px-2 py-1.5">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
                    Delta FRT
                  </p>
                  <p
                    className={`text-sm font-mono font-semibold ${
                      rec.expected_delta_frt >= 0
                        ? "text-green-600 dark:text-green-400"
                        : "text-red-600 dark:text-red-400"
                    }`}
                  >
                    {rec.expected_delta_frt >= 0 ? "+" : ""}
                    {rec.expected_delta_frt.toFixed(1)}m
                  </p>
                </div>
                <div className="rounded-md bg-muted/50 px-2 py-1.5">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
                    Delta P(NF)
                  </p>
                  <p
                    className={`text-sm font-mono font-semibold ${
                      rec.expected_delta_prob >= 0
                        ? "text-green-600 dark:text-green-400"
                        : "text-red-600 dark:text-red-400"
                    }`}
                  >
                    {rec.expected_delta_prob >= 0 ? "+" : ""}
                    {(rec.expected_delta_prob * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="rounded-md bg-muted/50 px-2 py-1.5">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
                    New FRT
                  </p>
                  <p className="text-sm font-mono font-semibold">
                    {rec.expected_frt.toFixed(0)}m
                  </p>
                </div>
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">
                {rec.rationale}
              </p>
            </div>
          ))}
        </CardContent>
      </Card>

      {combination && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">
              Combined Recommendation
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <ul className="space-y-1">
              {combination.actions.map((action, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <ArrowRight className="h-3.5 w-3.5 mt-0.5 text-primary shrink-0" />
                  {action}
                </li>
              ))}
            </ul>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="rounded-md bg-muted/50 px-2 py-1.5">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
                  Delta FRT
                </p>
                <p
                  className={`text-sm font-mono font-semibold ${
                    combination.expected_delta_frt >= 0
                      ? "text-green-600 dark:text-green-400"
                      : "text-red-600 dark:text-red-400"
                  }`}
                >
                  {combination.expected_delta_frt >= 0 ? "+" : ""}
                  {combination.expected_delta_frt.toFixed(1)}m
                </p>
              </div>
              <div className="rounded-md bg-muted/50 px-2 py-1.5">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
                  Delta P(NF)
                </p>
                <p
                  className={`text-sm font-mono font-semibold ${
                    combination.expected_delta_prob >= 0
                      ? "text-green-600 dark:text-green-400"
                      : "text-red-600 dark:text-red-400"
                  }`}
                >
                  {combination.expected_delta_prob >= 0 ? "+" : ""}
                  {(combination.expected_delta_prob * 100).toFixed(1)}%
                </p>
              </div>
              <div className="rounded-md bg-muted/50 px-2 py-1.5">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
                  Mode
                </p>
                <div className="mt-0.5">
                  <ModeBadge mode={combination.predicted_mode} className="text-xs py-0 px-1.5" />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {notes.length > 0 && (
        <div className="rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 p-3">
          {notes.map((note, i) => (
            <p key={i} className="text-sm text-blue-800 dark:text-blue-300">
              {note}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
