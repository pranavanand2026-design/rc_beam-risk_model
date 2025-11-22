"use client";

import { useState, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useFeatureMeta, useBeamList } from "@/hooks/useFeatureMeta";
import { api } from "@/lib/api";
import { FEATURE_GROUPS } from "@/lib/constants";
import type { BeamInput, GroundTruth, MetaResponse } from "@/lib/types";
import { Loader2, Database, Upload, PenLine } from "lucide-react";
import { BeamBrowser } from "./BeamBrowser";
import { CsvUploadZone } from "./CsvUploadZone";
import { SelectedBeamSummary } from "./SelectedBeamSummary";

interface InputPanelProps {
  onSubmit: (input: BeamInput) => void;
  isLoading: boolean;
  initialBeamId?: string | null;
}

type TabValue = "browse" | "csv" | "manual";

export function InputPanel({
  onSubmit,
  isLoading,
  initialBeamId,
}: InputPanelProps) {
  const { data: meta } = useFeatureMeta();
  const [tab, setTab] = useState<TabValue>(
    initialBeamId ? "browse" : "browse"
  );
  const [features, setFeatures] = useState<Record<string, number>>({});
  const [exposure, setExposure] = useState(90);
  const [margin, setMargin] = useState(10);
  const [selectedBeamId, setSelectedBeamId] = useState<string | null>(
    initialBeamId ?? null
  );
  const [groundTruth, setGroundTruth] = useState<GroundTruth | null>(null);
  const [sourceLabel, setSourceLabel] = useState<string | null>(null);
  const [featuresLoaded, setFeaturesLoaded] = useState(false);

  // Initialize defaults from meta
  useEffect(() => {
    if (meta && Object.keys(features).length === 0) {
      setFeatures({ ...meta.defaults });
    }
  }, [meta, features]);

  // Load beam from URL param
  useEffect(() => {
    if (initialBeamId && meta) {
      setSelectedBeamId(initialBeamId);
      setTab("browse");
      api.getBeam(initialBeamId).then((beam) => {
        setFeatures(beam.features);
        setGroundTruth(beam.ground_truth ?? null);
        setSourceLabel(`Dataset: ${initialBeamId}`);
        setFeaturesLoaded(true);
      }).catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialBeamId, meta]);

  const allFeatures = useMemo(() => {
    if (!meta) return [];
    return [...new Set([...meta.classifier_features, ...meta.regressor_features])];
  }, [meta]);

  const handleBeamSelect = (
    beamId: string,
    beamFeatures: Record<string, number>,
    gt?: GroundTruth | null
  ) => {
    setSelectedBeamId(beamId);
    setFeatures(beamFeatures);
    setGroundTruth(gt ?? null);
    setSourceLabel(`Dataset: ${beamId}`);
    setFeaturesLoaded(true);
  };

  const handleCsvLoaded = (
    csvFeatures: Record<string, number>,
    label?: string
  ) => {
    setFeatures(csvFeatures);
    setSelectedBeamId(null);
    setGroundTruth(null);
    setSourceLabel(`CSV: ${label ?? "uploaded"}`);
    setFeaturesLoaded(true);
  };

  const handleFeatureChange = (name: string, value: string) => {
    const num = parseFloat(value);
    if (!isNaN(num)) {
      setFeatures((prev) => ({ ...prev, [name]: num }));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      features,
      exposure_minutes: exposure,
      margin_minutes: margin,
    });
  };

  if (!meta) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Beam Parameters</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <Tabs
            value={tab}
            onValueChange={(v) => setTab(v as TabValue)}
          >
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="browse">
                <Database className="h-3.5 w-3.5 mr-1.5" />
                Browse
              </TabsTrigger>
              <TabsTrigger value="csv">
                <Upload className="h-3.5 w-3.5 mr-1.5" />
                Upload CSV
              </TabsTrigger>
              <TabsTrigger value="manual">
                <PenLine className="h-3.5 w-3.5 mr-1.5" />
                Manual
              </TabsTrigger>
            </TabsList>

            <TabsContent value="browse" className="mt-4">
              <BeamBrowser
                onSelect={handleBeamSelect}
                selectedBeamId={selectedBeamId}
              />
            </TabsContent>

            <TabsContent value="csv" className="mt-4">
              <CsvUploadZone
                onFeaturesLoaded={handleCsvLoaded}
                expectedFeatures={allFeatures}
                featureMeta={meta.features}
                defaults={meta.defaults}
              />
            </TabsContent>

            <TabsContent value="manual" className="mt-4">
              <div className="text-sm text-muted-foreground mb-3">
                Enter or adjust values for all model features.
              </div>
              <div className="space-y-4">
                {Object.entries(FEATURE_GROUPS).map(([group, feats]) => {
                  const relevant = feats.filter((f) =>
                    allFeatures.includes(f)
                  );
                  if (relevant.length === 0) return null;
                  return (
                    <FeatureGroup
                      key={group}
                      title={group}
                      features={relevant}
                      values={features}
                      meta={meta}
                      onChange={handleFeatureChange}
                    />
                  );
                })}
              </div>
            </TabsContent>
          </Tabs>

          {/* Summary when features loaded from browse or CSV */}
          {featuresLoaded && tab !== "manual" && sourceLabel && (
            <SelectedBeamSummary
              features={features}
              meta={meta}
              groundTruth={groundTruth}
              sourceLabel={sourceLabel}
              onEditClick={() => setTab("manual")}
            />
          )}

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="exposure">Fire exposure (min)</Label>
              <Input
                id="exposure"
                type="number"
                value={exposure}
                onChange={(e) =>
                  setExposure(parseFloat(e.target.value) || 0)
                }
                min={0}
                step={5}
              />
            </div>
            <div>
              <Label htmlFor="margin">Safety margin (min)</Label>
              <Input
                id="margin"
                type="number"
                value={margin}
                onChange={(e) =>
                  setMargin(parseFloat(e.target.value) || 0)
                }
                min={0}
                step={5}
              />
            </div>
          </div>

          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading && (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            )}
            Analyze Beam
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

function FeatureGroup({
  title,
  features,
  values,
  meta,
  onChange,
}: {
  title: string;
  features: readonly string[];
  values: Record<string, number>;
  meta: MetaResponse;
  onChange: (name: string, value: string) => void;
}) {
  return (
    <div>
      <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">
        {title}
      </h4>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {features.map((f) => {
          const fm = meta.features[f];
          const label = fm
            ? `${fm.label}${fm.unit ? ` (${fm.unit})` : ""}`
            : f;
          return (
            <div key={f}>
              <Label htmlFor={`feat-${f}`} className="text-xs">
                {label}
              </Label>
              <Input
                id={`feat-${f}`}
                type="number"
                value={values[f] ?? 0}
                onChange={(e) => onChange(f, e.target.value)}
                step="any"
                className="h-8 text-sm"
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
