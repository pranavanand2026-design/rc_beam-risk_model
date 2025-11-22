"use client";

import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { BeamInput } from "@/lib/types";

export function useAnalyze() {
  return useMutation({
    mutationFn: (input: BeamInput) => api.analyze(input),
    onSuccess: () => {
      toast.success("Analysis complete");
    },
    onError: (err: Error) => {
      toast.error(`Analysis failed: ${err.message}`);
    },
  });
}
