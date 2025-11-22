"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function useFeatureMeta() {
  return useQuery({
    queryKey: ["meta"],
    queryFn: api.getMeta,
    staleTime: Infinity,
  });
}

export function useBeamList() {
  return useQuery({
    queryKey: ["beams"],
    queryFn: api.listBeams,
    staleTime: Infinity,
  });
}

export function useBeamDetail(beamId: string | null) {
  return useQuery({
    queryKey: ["beam", beamId],
    queryFn: () => api.getBeam(beamId!),
    enabled: !!beamId,
    staleTime: Infinity,
  });
}
