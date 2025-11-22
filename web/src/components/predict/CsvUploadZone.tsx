"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, FileText, Download, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { parseBeamCsv, downloadCsvTemplate } from "@/lib/csv-utils";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import type { CsvParseResult, FeatureMeta } from "@/lib/types";

interface CsvUploadZoneProps {
  onFeaturesLoaded: (features: Record<string, number>, label?: string) => void;
  expectedFeatures: string[];
  featureMeta: Record<string, FeatureMeta>;
  defaults: Record<string, number>;
}

export function CsvUploadZone({
  onFeaturesLoaded,
  expectedFeatures,
  featureMeta,
  defaults,
}: CsvUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [result, setResult] = useState<CsvParseResult | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith(".csv")) {
        toast.error("Please upload a .csv file");
        return;
      }

      const parsed = await parseBeamCsv(
        file,
        expectedFeatures,
        featureMeta,
        defaults
      );
      setResult(parsed);

      if (parsed.missingColumns.length > 0) {
        toast.warning(
          `Missing columns: ${parsed.missingColumns.slice(0, 5).join(", ")}${parsed.missingColumns.length > 5 ? "..." : ""}`
        );
      }

      if (parsed.validRows === 0) {
        toast.error("No valid rows found in CSV");
        return;
      }

      if (parsed.validRows === 1 || (parsed.rows.length === 1 && parsed.rows[0].valid)) {
        const row = parsed.rows.find((r) => r.valid)!;
        onFeaturesLoaded(row.features, row.label);
        toast.success(`Loaded beam "${row.label}" from ${file.name}`);
      } else {
        toast.success(
          `Parsed ${parsed.validRows} valid rows from ${file.name}`
        );
      }
    },
    [expectedFeatures, featureMeta, defaults, onFeaturesLoaded]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = "";
  };

  const handleRowPick = (row: CsvParseResult["rows"][number]) => {
    onFeaturesLoaded(row.features, row.label);
    toast.success(`Loaded beam "${row.label}"`);
  };

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={cn(
          "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-8 cursor-pointer transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50"
        )}
      >
        <Upload className="h-8 w-8 text-muted-foreground" />
        <div className="text-center">
          <p className="text-sm font-medium">
            Drop a CSV file here or click to browse
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Expects columns matching model features ({expectedFeatures.length}{" "}
            features)
          </p>
        </div>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Multi-row picker */}
      {result && result.validRows > 1 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">
            <FileText className="inline h-3 w-3 mr-1" />
            {result.fileName} &mdash; {result.validRows} valid /{" "}
            {result.totalRows} total rows
          </p>
          <div className="max-h-48 overflow-y-auto rounded-md border divide-y">
            {result.rows.map((row, i) => (
              <button
                key={i}
                type="button"
                disabled={!row.valid}
                onClick={() => handleRowPick(row)}
                className={cn(
                  "flex items-center gap-2 w-full px-3 py-2 text-left text-sm transition-colors",
                  row.valid
                    ? "hover:bg-accent cursor-pointer"
                    : "opacity-50 cursor-not-allowed"
                )}
              >
                {row.valid ? (
                  <CheckCircle2 className="h-3.5 w-3.5 text-green-600 shrink-0" />
                ) : (
                  <AlertCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
                )}
                <span className="font-mono text-xs truncate">{row.label}</span>
                {!row.valid && (
                  <span className="text-xs text-destructive ml-auto truncate max-w-[50%]">
                    {row.errors[0]}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Template download */}
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="text-xs"
        onClick={() => downloadCsvTemplate(expectedFeatures, featureMeta, defaults)}
      >
        <Download className="h-3 w-3 mr-1" />
        Download CSV template
      </Button>
    </div>
  );
}
