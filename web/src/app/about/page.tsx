import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Flame,
  BrainCircuit,
  Target,
  Shield,
  BarChart3,
  BookOpen,
  AlertTriangle,
} from "lucide-react";

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12">
      {/* Hero */}
      <div className="mb-12">
        <Badge variant="outline" className="mb-4">
          <BookOpen className="h-3 w-3 mr-1" />
          Research Context
        </Badge>
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight mb-4">
          Interpretable ML for Fire-Exposed{" "}
          <span className="text-primary">FRP-Strengthened RC Beams</span>
        </h1>
        <p className="text-muted-foreground text-base sm:text-lg leading-relaxed max-w-3xl">
          This tool implements a hybrid machine learning framework that jointly
          predicts failure mode and fire resistance time (FRT) for reinforced
          concrete beams strengthened with externally bonded FRP under fire
          loading.
        </p>
      </div>

      <div className="space-y-8">
        {/* Problem Statement */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-start gap-3 mb-4">
              <div className="rounded-lg bg-destructive/10 p-2">
                <Flame className="h-5 w-5 text-destructive" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">The Problem</h2>
                <p className="text-sm text-muted-foreground">
                  Why FRP-strengthened beams need special attention under fire
                </p>
              </div>
            </div>
            <div className="space-y-3 text-sm leading-relaxed">
              <p>
                FRP-strengthened RC beams are vulnerable under fire due to
                adhesive softening, interfacial debonding, and thermal gradients
                that couple serviceability and ultimate limit states. As
                temperatures rise, the adhesive bonding FRP to concrete degrades
                near its glass transition temperature (T<sub>g</sub>), causing
                load redistribution, large deflections, and potential strength
                collapse.
              </p>
              <p>
                While prior experimental and numerical work has clarified the
                underlying mechanisms, and recent ML efforts can predict FRT,
                there remains a gap in <strong>mode-level prediction</strong>{" "}
                with interpretable, safety-aware outputs suitable for design
                decisions. Eurocode EN 1992-1-2 provides system-level guidance
                but cannot fully capture project-specific geometries, anchorage
                schemes, and insulation details.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Approach */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-start gap-3 mb-4">
              <div className="rounded-lg bg-primary/10 p-2">
                <BrainCircuit className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">Our Approach</h2>
                <p className="text-sm text-muted-foreground">
                  Hybrid, safety-calibrated ML framework
                </p>
              </div>
            </div>
            <div className="space-y-3 text-sm leading-relaxed">
              <p>
                We develop an interpretable hybrid ML framework that jointly
                predicts failure mode (Deflection Failure, Strength Failure, No
                Failure) and FRT from raw physical features. The pipeline has two
                components:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 my-4">
                <div className="rounded-lg border p-3 space-y-1">
                  <p className="font-medium text-sm">Failure Mode Classifier</p>
                  <p className="text-xs text-muted-foreground">
                    Blends XGBoost + Targeted SMOTE with an LDAM-DRW XGBoost leg
                    trained with Borderline-SMOTE. Probabilities are
                    isotonic-calibrated and per-class thresholded to enforce a
                    safety-first bias.
                  </p>
                </div>
                <div className="rounded-lg border p-3 space-y-1">
                  <p className="font-medium text-sm">FRT Regressor</p>
                  <p className="text-xs text-muted-foreground">
                    A LightGBM model that predicts fire resistance time in
                    minutes, trained on the Bhatt-Kodur-Naser dataset with
                    physical features as inputs.
                  </p>
                </div>
              </div>
              <p>
                SHAP analysis identifies insulation thickness/depth, concrete
                cover, FRP area, and load ratio as dominant drivers, consistent
                with fire mechanics and Eurocode EN 1992-1-2 framing.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Performance */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-start gap-3 mb-4">
              <div className="rounded-lg bg-green-500/10 p-2">
                <BarChart3 className="h-5 w-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">Model Performance</h2>
                <p className="text-sm text-muted-foreground">
                  Held-out split from the Bhatt-Kodur-Naser dataset
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: "Accuracy", value: "~0.906" },
                { label: "Balanced Acc.", value: "~0.824" },
                { label: "Macro-F1", value: "~0.818" },
                { label: "FRT MAE", value: "~10.3 min" },
              ].map((m) => (
                <div
                  key={m.label}
                  className="rounded-lg border bg-muted/30 p-3 text-center"
                >
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wide">
                    {m.label}
                  </p>
                  <p className="text-lg font-mono font-semibold mt-1">
                    {m.value}
                  </p>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-3">
              FRT regressor achieves R&#178; ~ 0.93. Classifier calibrated with
              isotonic regression for reliable probability estimates.
            </p>
          </CardContent>
        </Card>

        {/* Input Features */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-start gap-3 mb-4">
              <div className="rounded-lg bg-amber-500/10 p-2">
                <Target className="h-5 w-5 text-amber-600 dark:text-amber-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">Input Features</h2>
                <p className="text-sm text-muted-foreground">
                  Up to 16 physical parameters across 5 categories
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 text-sm">
              {[
                {
                  group: "Geometry",
                  items:
                    "Span length (L), cross-section area (Ac), concrete cover (Cc)",
                },
                {
                  group: "Materials",
                  items:
                    "Concrete strength (fc), steel yield (fy), steel modulus (Es), FRP strength (fu), FRP modulus (Efrp), glass transition temp (Tg)",
                },
                {
                  group: "Reinforcement",
                  items: "Steel area (As), FRP area (Af)",
                },
                {
                  group: "Protection",
                  items:
                    "Soffit insulation thickness (tins), side insulation depth (hi), insulation conductivity (kins), thermal resistance (rinscins)",
                },
                {
                  group: "Loading",
                  items: "Applied load (Ld), load ratio (LR)",
                },
              ].map((g) => (
                <div key={g.group}>
                  <p className="font-medium">{g.group}</p>
                  <p className="text-muted-foreground text-xs">{g.items}</p>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-4">
              Key adjustable parameters for design improvement: concrete cover
              (Cc), insulation thickness (tins), side insulation depth (hi), load
              ratio (LR), steel area (As), and FRP area (Af).
            </p>
          </CardContent>
        </Card>

        {/* Action Plans */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-start gap-3 mb-4">
              <div className="rounded-lg bg-primary/10 p-2">
                <Shield className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">
                  Design Recommendations
                </h2>
                <p className="text-sm text-muted-foreground">
                  Safety-aware, interpretable action plans
                </p>
              </div>
            </div>
            <div className="space-y-3 text-sm leading-relaxed">
              <p>
                When a beam is at risk, the system generates ranked
                single-parameter recommendations and an optimal multi-parameter
                combination. Each recommendation shows the expected change in FRT
                and probability of no failure, based on what-if predictions.
                Target values are guided by safe-beam reference percentiles and
                bounded by practical Eurocode-aligned ranges.
              </p>
              <p>
                The analysis also includes an approximate check against
                EN 1992-1-2 fire design envelopes (R60/R90/R120), evaluating
                minimum cover, insulation depth, and maximum load ratio
                requirements.
              </p>
            </div>
          </CardContent>
        </Card>

        <Separator />

        {/* Disclaimer */}
        <div className="flex gap-3 rounded-lg border border-amber-200 dark:border-amber-900 bg-amber-50 dark:bg-amber-950/30 p-4">
          <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-2 text-sm">
            <p className="font-medium text-amber-800 dark:text-amber-300">
              Important Limitations
            </p>
            <p className="text-amber-700 dark:text-amber-400/90 leading-relaxed">
              Because &ldquo;No Failure&rdquo; is rare (~2-3% of samples),
              interpretability provides directional risk reduction rather than a
              prescriptive &ldquo;design-to-flip&rdquo; guarantee. This tool
              supports engineering judgment; it does not replace it. Full
              structural analysis and professional review remain essential for
              final design decisions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
