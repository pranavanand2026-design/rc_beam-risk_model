"use client";

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FeatureSliderPanel } from "@/components/playground/FeatureSliderPanel";
import { LiveResultsPanel } from "@/components/playground/LiveResultsPanel";
import { useFeatureMeta, useBeamList } from "@/hooks/useFeatureMeta";
import { api } from "@/lib/api";
import type { PlaygroundResponse } from "@/lib/types";
import { Loader2, Search } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const MAX_BEAM_RESULTS = 30;

export default function PlaygroundClient() {
  const { data: meta } = useFeatureMeta();
  const { data: beamList } = useBeamList();

  const [selectedBeam, setSelectedBeam] = useState<string>("");
  const [beamSearch, setBeamSearch] = useState("");
  const [beamDropdownOpen, setBeamDropdownOpen] = useState(false);
  const [baseline, setBaseline] = useState<Record<string, number>>({});
  const [values, setValues] = useState<Record<string, number>>({});
  const [result, setResult] = useState<PlaygroundResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [exposure, setExposure] = useState(90);
  const [margin, setMargin] = useState(10);
  const abortRef = useRef<AbortController | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const filteredBeams = useMemo(() => {
    if (!beamList) return [];
    const q = beamSearch.trim().toLowerCase();
    const matches = q
      ? beamList.beams.filter((b) => b.id.toLowerCase().includes(q))
      : beamList.beams;
    return matches.slice(0, MAX_BEAM_RESULTS);
  }, [beamList, beamSearch]);

  useEffect(() => {
    if (meta && Object.keys(baseline).length === 0) {
      setBaseline({ ...meta.defaults });
      setValues({ ...meta.defaults });
    }
  }, [meta, baseline]);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setBeamDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const fetchPrediction = useCallback(
    async (
      feats: Record<string, number>,
      base: Record<string, number>,
      exp: number,
      mar: number
    ) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setIsLoading(true);
      try {
        const data = await api.playground(
          {
            features: feats,
            baseline_features: base,
            exposure_minutes: exp,
            margin_minutes: mar,
          },
          controller.signal
        );
        setResult(data);
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") return;
      } finally {
        if (!controller.signal.aborted) setIsLoading(false);
      }
    },
    []
  );

  const handleBeamSelect = async (beamId: string) => {
    setSelectedBeam(beamId);
    setBeamSearch(beamId);
    setBeamDropdownOpen(false);
    try {
      const beam = await api.getBeam(beamId);
      setBaseline(beam.features);
      setValues(beam.features);
      fetchPrediction(beam.features, beam.features, exposure, margin);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      toast.error(`Failed to load beam: ${msg}`);
    }
  };

  const handleSliderChange = useCallback(
    (feature: string, value: number) => {
      setValues((prev) => ({ ...prev, [feature]: value }));
    },
    []
  );

  const handleSliderCommit = useCallback(
    (feature: string, value: number) => {
      setValues((prev) => {
        const next = { ...prev, [feature]: value };
        fetchPrediction(next, baseline, exposure, margin);
        return next;
      });
    },
    [baseline, exposure, margin, fetchPrediction]
  );

  const handleReset = useCallback(() => {
    setValues({ ...baseline });
    fetchPrediction(baseline, baseline, exposure, margin);
  }, [baseline, exposure, margin, fetchPrediction]);

  if (!meta) {
    return (
      <div className="flex items-center justify-center min-h-[40vh]">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <>
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold">Design Playground</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Adjust parameters with sliders and see predictions update live
          </p>
        </div>
        <div ref={dropdownRef} className="relative w-56">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground pointer-events-none" />
          <Input
            placeholder="Search beams..."
            value={beamSearch}
            onChange={(e) => {
              setBeamSearch(e.target.value);
              setBeamDropdownOpen(true);
            }}
            onFocus={() => setBeamDropdownOpen(true)}
            className="pl-8 h-9"
          />
          {beamDropdownOpen && filteredBeams.length > 0 && (
            <div className="absolute z-50 top-full mt-1 w-full rounded-md border bg-popover shadow-md max-h-56 overflow-y-auto">
              {filteredBeams.map((b) => (
                <button
                  key={b.id}
                  type="button"
                  onClick={() => handleBeamSelect(b.id)}
                  className={cn(
                    "w-full text-left px-3 py-1.5 text-sm hover:bg-accent transition-colors",
                    selectedBeam === b.id && "bg-accent font-medium"
                  )}
                >
                  {b.id}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-4">
          <Card>
            <CardContent className="pt-6">
              <FeatureSliderPanel
                adjustableFeatures={meta.adjustable_features}
                featureMeta={meta.features}
                values={values}
                baseline={baseline}
                onChange={handleSliderChange}
                onCommit={handleSliderCommit}
                onReset={handleReset}
              />
              <div className="mt-6 grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs">Exposure (min)</Label>
                  <Input
                    type="number"
                    value={exposure}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value) || 0;
                      setExposure(v);
                      fetchPrediction(values, baseline, v, margin);
                    }}
                    min={0}
                    step={5}
                    className="h-8 text-sm"
                  />
                </div>
                <div>
                  <Label className="text-xs">Margin (min)</Label>
                  <Input
                    type="number"
                    value={margin}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value) || 0;
                      setMargin(v);
                      fetchPrediction(values, baseline, exposure, v);
                    }}
                    min={0}
                    step={5}
                    className="h-8 text-sm"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-8">
          <Card>
            <CardContent className="pt-6">
              <LiveResultsPanel
                result={result}
                isLoading={isLoading}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
