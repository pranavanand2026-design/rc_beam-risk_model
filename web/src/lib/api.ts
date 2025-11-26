import type {
  AnalyzeResponse,
  BeamDetailResponse,
  BeamInput,
  BeamListResponse,
  MetaResponse,
  PlaygroundInput,
  PlaygroundResponse,
  PredictionResponse,
} from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://rcbeam-riskmodel-production.up.railway.app";

async function fetchJSON<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  getMeta: () => fetchJSON<MetaResponse>("/api/v1/meta"),

  listBeams: () => fetchJSON<BeamListResponse>("/api/v1/meta/beams"),

  getBeam: (id: string) =>
    fetchJSON<BeamDetailResponse>(
      `/api/v1/meta/beams/${encodeURIComponent(id)}`
    ),

  predict: (input: BeamInput) =>
    fetchJSON<PredictionResponse>("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify(input),
    }),

  analyze: (input: BeamInput) =>
    fetchJSON<AnalyzeResponse>("/api/v1/analyze", {
      method: "POST",
      body: JSON.stringify(input),
    }),

  playground: (input: PlaygroundInput, signal?: AbortSignal) =>
    fetchJSON<PlaygroundResponse>("/api/v1/playground", {
      method: "POST",
      body: JSON.stringify(input),
      signal,
    }),
};
