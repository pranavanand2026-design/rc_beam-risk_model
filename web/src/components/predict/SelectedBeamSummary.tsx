"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FEATURE_GROUPS } from "@/lib/constants";
import type { GroundTruth, MetaResponse } from "@/lib/types";
import { Pencil } from "lucide-react";

const GROUP_SHORT: Record<string, string> = {
  Geometry: "Geom",
  Materials: "Mat",
  Reinforcement: "Reinf",
  Protection: "Prot",
  Loading: "Load",
};

interface SelectedBeamSummaryProps {
  features: Record<string, number>;
  meta: MetaResponse;
  groundTruth?: GroundTruth | null;
  sourceLabel: string;
  onEditClick: () => void;
}

export function SelectedBeamSummary({
  features,
  meta,
  groundTruth,
  sourceLabel,
  onEditClick,
}: SelectedBeamSummaryProps) {
  const allFeatures = [
    ...new Set([...meta.classifier_features, ...meta.regressor_features]),
  ];

  return (
    <div className="rounded-lg border bg-muted/30 p-3 space-y-2.5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-muted-foreground">
          {sourceLabel}
        </p>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="h-6 text-xs px-2"
          onClick={onEditClick}
        >
          <Pencil className="h-3 w-3 mr-1" />
          Edit values
        </Button>
      </div>

      {groundTruth && (groundTruth.mode || groundTruth.frt) && (
        <div className="flex items-center gap-2">
          {groundTruth.mode && (
            <Badge variant="outline" className="text-xs">
              Actual: {groundTruth.mode}
            </Badge>
          )}
          {groundTruth.frt != null && (
            <Badge variant="outline" className="text-xs">
              Actual FRT: {groundTruth.frt} min
            </Badge>
          )}
        </div>
      )}

      <div className="space-y-1.5">
        {Object.entries(FEATURE_GROUPS).map(([group, feats]) => {
          const relevant = feats.filter((f) => allFeatures.includes(f));
          if (relevant.length === 0) return null;
          return (
            <div key={group}>
              <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                {GROUP_SHORT[group] ?? group}
              </span>
              <div className="flex flex-wrap gap-1 mt-0.5">
                {relevant.map((f) => {
                  const fm = meta.features[f];
                  const unit = fm?.unit ? ` ${fm.unit}` : "";
                  const val = features[f];
                  return (
                    <span
                      key={f}
                      className="inline-flex items-center rounded bg-background border px-1.5 py-0.5 text-[11px] font-mono whitespace-nowrap"
                    >
                      {fm?.label ?? f}: {val != null ? val : "?"}
                      {unit}
                    </span>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
