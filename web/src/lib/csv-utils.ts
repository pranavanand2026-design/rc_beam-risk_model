import Papa from "papaparse";
import type { CsvParseResult, ParsedBeamRow, FeatureMeta } from "./types";

export function parseBeamCsv(
  file: File,
  expectedFeatures: string[],
  featureMeta: Record<string, FeatureMeta>,
  defaults: Record<string, number>
): Promise<CsvParseResult> {
  return new Promise((resolve) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete(results) {
        const headers = results.meta.fields ?? [];
        const headerMap = new Map(
          headers.map((h) => [h.toLowerCase().trim(), h])
        );

        const missingColumns = expectedFeatures.filter(
          (f) => !headerMap.has(f.toLowerCase())
        );

        const rows: ParsedBeamRow[] = (
          results.data as Record<string, unknown>[]
        ).map((row, i) => {
          const features: Record<string, number> = {};
          const errors: string[] = [];

          for (const feat of expectedFeatures) {
            const csvCol = headerMap.get(feat.toLowerCase());
            const raw = csvCol !== undefined ? row[csvCol] : undefined;

            if (raw === undefined || raw === null || raw === "") {
              if (defaults[feat] !== undefined) {
                features[feat] = defaults[feat];
              } else {
                errors.push(`Missing "${feat}"`);
              }
            } else {
              const num = typeof raw === "number" ? raw : parseFloat(String(raw));
              if (!isFinite(num)) {
                errors.push(`"${feat}" is not a valid number`);
              } else {
                features[feat] = num;
              }
            }
          }

          const labelCol = headerMap.get("label") ?? headerMap.get("id") ?? headerMap.get("beam_id");
          const label = labelCol && row[labelCol]
            ? String(row[labelCol])
            : `Row ${i + 1}`;

          return { label, features, valid: errors.length === 0, errors };
        });

        resolve({
          fileName: file.name,
          rows,
          totalRows: rows.length,
          validRows: rows.filter((r) => r.valid).length,
          missingColumns,
        });
      },
    });
  });
}

export function downloadCsvTemplate(
  features: string[],
  featureMeta: Record<string, FeatureMeta>,
  defaults: Record<string, number>
) {
  const header = ["label", ...features];
  const exampleRow = [
    "example_beam",
    ...features.map((f) => String(defaults[f] ?? 0)),
  ];

  const csv = [header.join(","), exampleRow.join(",")].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "beam_template.csv";
  a.click();
  URL.revokeObjectURL(url);
}
