"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import Link from "next/link";
import { ArrowDown, FlaskConical, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { InputPanel } from "@/components/predict/InputPanel";
import { ResultsPanel } from "@/components/predict/ResultsPanel";
import { useAnalyze } from "@/hooks/useAnalyze";

function HomePage() {
  const searchParams = useSearchParams();
  const beamParam = searchParams.get("beam");
  const analyze = useAnalyze();

  return (
    <div>
      {/* Hero Section */}
      <section className="relative overflow-hidden border-b bg-gradient-to-b from-primary/5 to-background">
        <div className="mx-auto max-w-6xl px-4 py-16 md:py-24">
          <div className="text-center max-w-2xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="inline-flex items-center gap-2 rounded-full border bg-background/80 px-3 py-1 text-sm text-muted-foreground mb-6">
              <FlaskConical className="h-3.5 w-3.5" />
              ML-Powered Fire Design
            </div>
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight">
              RC Beam Fire{" "}
              <span className="text-primary">Design Studio</span>
            </h1>
            <p className="mt-4 text-muted-foreground text-base sm:text-lg leading-relaxed">
              Pick a beam from the built-in dataset, upload your own CSV, or
              enter values manually. Get failure mode predictions, fire
              resistance estimates, and actionable design recommendations.
            </p>
            <div className="mt-6 flex items-center justify-center gap-3">
              <Button
                size="lg"
                onClick={() =>
                  document
                    .getElementById("predict")
                    ?.scrollIntoView({ behavior: "smooth" })
                }
              >
                Browse Beams
                <ArrowDown className="ml-2 h-4 w-4" />
              </Button>
              <Button variant="outline" size="lg" asChild>
                <Link href="/playground">Open Playground</Link>
              </Button>
            </div>
            <p className="mt-4 text-xs text-muted-foreground">
              Press{" "}
              <kbd className="rounded border bg-muted px-1.5 py-0.5 text-xs font-mono">
                Cmd+K
              </kbd>{" "}
              to search beams instantly
            </p>
          </div>
        </div>
      </section>

      {/* Quick Predict Section */}
      <section id="predict" className="mx-auto max-w-6xl px-4 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <InputPanel
              onSubmit={(input) => analyze.mutate(input)}
              isLoading={analyze.isPending}
              initialBeamId={beamParam}
            />
          </div>
          <div>
            {analyze.data && <ResultsPanel result={analyze.data} />}
            {!analyze.data && !analyze.isPending && (
              <div className="flex items-center justify-center h-64 text-muted-foreground text-sm">
                <div className="text-center space-y-2">
                  <Search className="h-8 w-8 mx-auto text-muted-foreground/50" />
                  <p>Submit the form to see analysis results</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

export default function Page() {
  return (
    <Suspense>
      <HomePage />
    </Suspense>
  );
}
