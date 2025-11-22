"use client";

import { useState, useMemo } from "react";
import { Input } from "@/components/ui/input";
import { useBeamList } from "@/hooks/useFeatureMeta";
import { api } from "@/lib/api";
import type { GroundTruth } from "@/lib/types";
import { Search, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface BeamBrowserProps {
  onSelect: (
    beamId: string,
    features: Record<string, number>,
    groundTruth?: GroundTruth | null
  ) => void;
  selectedBeamId: string | null;
}

export function BeamBrowser({ onSelect, selectedBeamId }: BeamBrowserProps) {
  const { data: beamList, isLoading } = useBeamList();
  const [search, setSearch] = useState("");
  const [loadingId, setLoadingId] = useState<string | null>(null);

  const MAX_VISIBLE = 50;

  const filtered = useMemo(() => {
    if (!beamList) return [];
    const q = search.trim().toLowerCase();
    const matches = q
      ? beamList.beams.filter((b) => b.id.toLowerCase().includes(q))
      : beamList.beams;
    return matches.slice(0, MAX_VISIBLE);
  }, [beamList, search]);

  const totalMatches = useMemo(() => {
    if (!beamList) return 0;
    const q = search.trim().toLowerCase();
    return q
      ? beamList.beams.filter((b) => b.id.toLowerCase().includes(q)).length
      : beamList.beams.length;
  }, [beamList, search]);

  const handleClick = async (beamId: string) => {
    setLoadingId(beamId);
    try {
      const beam = await api.getBeam(beamId);
      onSelect(beamId, beam.features, beam.ground_truth);
    } catch {
      // keep current state
    } finally {
      setLoadingId(null);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
        <Loader2 className="h-4 w-4 animate-spin mr-2" />
        Loading beams...
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="relative">
        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search beams..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-8 h-9"
        />
      </div>

      <p className="text-xs text-muted-foreground">
        Showing {filtered.length} of {totalMatches} beams
        {totalMatches > MAX_VISIBLE && !search.trim() && " — type to search"}
      </p>

      <div className="max-h-72 overflow-y-auto rounded-md border p-1">
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
          {filtered.map((b) => (
            <button
              key={b.id}
              type="button"
              onClick={() => handleClick(b.id)}
              disabled={loadingId !== null}
              className={cn(
                "relative rounded-md border px-2.5 py-2 text-left text-sm transition-colors hover:bg-accent",
                selectedBeamId === b.id
                  ? "ring-2 ring-primary bg-primary/5 border-primary/30"
                  : "border-border",
                loadingId === b.id && "opacity-70"
              )}
            >
              {loadingId === b.id && (
                <Loader2 className="absolute right-1.5 top-1.5 h-3 w-3 animate-spin" />
              )}
              <span className="font-mono text-xs">{b.id}</span>
            </button>
          ))}
        </div>

        {filtered.length === 0 && (
          <p className="text-center text-sm text-muted-foreground py-6">
            No beams match &quot;{search}&quot;
          </p>
        )}
      </div>
    </div>
  );
}
