"use client";

import { Badge } from "@/components/ui/badge";
import { MODE_BG_COLORS } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface ModeBadgeProps {
  mode: string;
  className?: string;
}

export function ModeBadge({ mode, className }: ModeBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={cn(
        "text-sm py-1 px-3",
        MODE_BG_COLORS[mode] || "bg-muted text-muted-foreground",
        className
      )}
    >
      {mode}
    </Badge>
  );
}
