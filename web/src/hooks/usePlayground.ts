"use client";

import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { PlaygroundInput } from "@/lib/types";

export function usePlayground() {
  return useMutation({
    mutationFn: (input: PlaygroundInput) => api.playground(input),
  });
}
