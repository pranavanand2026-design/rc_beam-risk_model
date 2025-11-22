export const FEATURE_GROUPS = {
  Geometry: ["L", "Ac", "Cc"],
  Materials: ["fc", "fy", "Es", "fu", "Efrp", "Tg"],
  Reinforcement: ["As", "Af"],
  Protection: ["tins", "hi", "kins", "rinscins"],
  Loading: ["Ld", "LR"],
} as const;

export const ADJUSTABLE_FEATURE_ORDER = [
  "Cc",
  "tins",
  "hi",
  "LR",
  "As",
  "Af",
] as const;

export const MODE_COLORS: Record<string, string> = {
  "No Failure": "#22c55e",
  "Strength Failure": "#ef4444",
  "Deflection Failure": "#f59e0b",
};

export const MODE_BG_COLORS: Record<string, string> = {
  "No Failure": "bg-green-500/10 text-green-700 dark:text-green-400",
  "Strength Failure": "bg-red-500/10 text-red-700 dark:text-red-400",
  "Deflection Failure":
    "bg-amber-500/10 text-amber-700 dark:text-amber-400",
};
