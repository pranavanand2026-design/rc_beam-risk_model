"use client";

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: number | string;
  suffix?: string;
  delta?: number;
  deltaSuffix?: string;
  className?: string;
  decimals?: number;
}

export function MetricCard({
  label,
  value,
  suffix,
  delta,
  deltaSuffix = "",
  className,
  decimals = 1,
}: MetricCardProps) {
  const isPositive = delta !== undefined && delta >= 0;

  return (
    <Card className={cn("", className)}>
      <CardContent className="pt-4 pb-3 px-4">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
          {label}
        </p>
        <div className="mt-1 flex items-baseline gap-1">
          <span className="text-2xl font-bold tabular-nums">
            {typeof value === "number" ? value.toFixed(decimals) : value}
          </span>
          {suffix && (
            <span className="text-sm text-muted-foreground">{suffix}</span>
          )}
        </div>
        {delta !== undefined && (
          <p
            className={cn(
              "mt-1 text-xs font-medium",
              isPositive
                ? "text-green-600 dark:text-green-400"
                : "text-red-600 dark:text-red-400"
            )}
          >
            {isPositive ? "+" : ""}
            {delta.toFixed(decimals)}
            {deltaSuffix}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
