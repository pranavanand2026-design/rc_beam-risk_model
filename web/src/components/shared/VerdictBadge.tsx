"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface VerdictBadgeProps {
  gap: number;
  className?: string;
}

export function VerdictBadge({ gap, className }: VerdictBadgeProps) {
  const safe = gap >= 0;

  return (
    <Badge
      variant="outline"
      className={cn(
        "text-sm py-1 px-3",
        safe
          ? "border-green-500/50 bg-green-500/10 text-green-700 dark:text-green-400"
          : "border-red-500/50 bg-red-500/10 text-red-700 dark:text-red-400",
        className
      )}
    >
      {safe ? `Meets requirement (+${gap.toFixed(1)} min)` : `At risk (${gap.toFixed(1)} min)`}
    </Badge>
  );
}
