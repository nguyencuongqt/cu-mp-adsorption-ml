import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "G:/My Drive/Cu-MP LLM prediction/cu_mp_adsorption";
const workbookPath = `${root}/data/raw_data_literature_master.xlsx`;
const appendPath = `${root}/Manuscript/Submission_Env pollution/Revision/Digitization_next_5_primary_isotherm_papers/finalized_data/master_append_rows_priority5.csv`;
const previewPath = `${root}/tmp/finalize_priority5/qe_model_cu_appended_preview.png`;

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === '"') {
      if (inQuotes && text[i + 1] === '"') { field += '"'; i += 1; }
      else inQuotes = !inQuotes;
    } else if (ch === "," && !inQuotes) {
      row.push(field); field = "";
    } else if (ch === "\n" && !inQuotes) {
      row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = "";
    } else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  const [headers, ...values] = rows.filter((r) => r.some((v) => v !== ""));
  return values.map((r) => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}

function num(value) {
  if (value === "" || value === "nan" || value === "NaN") return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`Expected a number, got ${value}`);
  return parsed;
}

function seriesLabel(row) {
  const temperature = row.Temp === "" ? "temperature not reported" : `${row.Temp} °C`;
  const state = row.AgS === "Virgin" ? "virgin" : row.ageing_detail;
  if (row.source_id === "Zhou_2022_WaterRes") return `PE ${row.series_id.replace(/.*_/, "")} | virgin`;
  return `${row.ReT} | ${state} | ${temperature}`;
}

const appendRows = parseCsv(await fs.readFile(appendPath, "utf8"));
if (appendRows.length !== 106) throw new Error(`Expected 106 reviewed rows, found ${appendRows.length}.`);
const included = appendRows.filter((row) => row.model_include === "True").length;
if (included !== 76) throw new Error(`Expected 76 included rows, found ${included}.`);

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const dataSheet = workbook.worksheets.getItem("qe_model_cu");
const metaSheet = workbook.worksheets.getItem("literature_metadata");
const readmeSheet = workbook.worksheets.getItem("README");
const validationSheet = workbook.worksheets.getItem("validation_summary");
const dataTable = dataSheet.tables.getItemAt(0);
const metaTable = metaSheet.tables.getItemAt(0);

const dataRows = appendRows.map((row, index) => [
  row.source_id,
  row.source_author_year,
  row.first_author,
  num(row.year),
  row.doi,
  "verified_original_pdf_figure_and_qc",
  2927 + index,
  row.model_include === "True",
  row.model_include === "True" ? null : "methods_figure_concentration_conflict",
  row.AgS,
  row.ReT,
  row.AdC,
  num(row.Temp),
  num(row.pH),
  num(row.rpm),
  num(row.Ce),
  num(row.qe),
  row.series_id,
  seriesLabel(row),
  row.ageing_type || null,
  row.ageing_detail || null,
]);
await dataTable.rows.add(null, dataRows);

await metaTable.rows.add(null, [
  ["Yu_2026", "(Yu et al., 2026)", "Yu", 2026, "Yu_2026_groundwater_PP_PE_UV_Cu", "10.1016/j.cej.2026.175301", 42, 42, "Cu (II)", "PP"],
  ["Lin_2023", "(Lin et al., 2023)", "Lin", 2023, "Lin_2023_conventional_biodegradable_bags_Cu", "10.1016/j.chemosphere.2023.138920", 12, 12, "Cu (II)", "PE; PLA"],
  ["Chen_2022", "(Chen et al., 2022)", "Chen", 2022, "Chen_2022_PS_UV_Cu", "10.1016/j.ecoenv.2022.113292", 30, 0, "Cu (II)", "PS"],
  ["Zhou_2022_WaterRes", "(Zhou et al., 2022)", "Zhou", 2022, "Zhou_2022_PE_size_Cu", "10.1016/j.watres.2022.118354", 12, 12, "Cu (II)", "PE"],
  ["Feng_2025", "(Feng et al., 2025)", "Feng", 2025, "Feng_2025_PLA_UV_Cu", "10.1016/j.jece.2025.117033", 10, 10, "Cu (II)", "PLA"],
]);

readmeSheet.getRange("B3").values = [["This workbook retains 973 Cu candidate rows for traceability; 859 rows with model_include=TRUE are used in data/dataset.csv and the current ML workflow."]];
readmeSheet.getRange("B4").values = [["qe_model_cu contains source metadata, model inclusion flags, model predictors AgS, ReT, AdC, Temp, pH, rpm, Ce, and target qe for 29 candidate literature studies (27 included in the model)."]];
readmeSheet.getRange("B6").values = [["data/dataset.csv contains the 859 rows with model_include=TRUE. The 114 QC-excluded candidate rows remain documented here with exclusion_reason values."]];
readmeSheet.getRange("B7").values = [["All 29 Cu candidate studies have DOI values. Mao_2020 and Zou_2020 were completed by DOI lookup because their PDFs are not currently present in data/papers."]];

validationSheet.getRange("B2:D2").values = [[973, "pass", "All Cu candidate rows, including review exclusions, are retained in qe_model_cu."]];
validationSheet.getRange("B3:D3").values = [[859, "pass", "Rows with model_include=TRUE; synchronized to data/dataset.csv and data/dataset_grouped.csv."]];
validationSheet.getRange("B4:D4").values = [[27, "pass", "Distinct source_id values represented in the model-ready subset after Priority-1 finalization."]];
validationSheet.getRange("B9:D9").values = [["complete", "info", "Priority-1 digitization appended: 76 included rows; 30 unresolved Chen_2022 rows retained as exclusions."]];

workbook.recalculate();
const check = await workbook.inspect({
  kind: "table",
  sheetId: "qe_model_cu",
  range: "A869:U974",
  include: "values,formulas",
  tableMaxRows: 106,
  tableMaxCols: 21,
  maxChars: 18000,
});
console.log(check.ndjson);
const metaCheck = await workbook.inspect({
  kind: "table",
  sheetId: "literature_metadata",
  range: "A25:J30",
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 10,
  maxChars: 6000,
});
console.log(metaCheck.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan after Priority-1 append",
});
console.log(errors.ndjson);
await fs.mkdir(`${root}/tmp/finalize_priority5`, { recursive: true });
const preview = await workbook.render({ sheetName: "qe_model_cu", range: "A869:U890", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(JSON.stringify({ status: "saved", rowsAdded: dataRows.length, included, workbookPath }));
