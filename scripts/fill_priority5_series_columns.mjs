import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "G:/My Drive/Cu-MP LLM prediction/cu_mp_adsorption";
const workbookPath = `${root}/data/raw_data_literature_master.xlsx`;
const appendPath = `${root}/Manuscript/Submission_Env pollution/Revision/Digitization_next_5_primary_isotherm_papers/finalized_data/master_append_rows_priority5.csv`;

function parseCsv(text) {
  const rows = []; let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === '"') { if (quoted && text[i + 1] === '"') { field += '"'; i += 1; } else quoted = !quoted; }
    else if (ch === "," && !quoted) { row.push(field); field = ""; }
    else if (ch === "\n" && !quoted) { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  const [headers, ...values] = rows.filter((r) => r.some((v) => v !== ""));
  return values.map((r) => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""])));
}

function label(row) {
  const temp = row.Temp === "" ? "temperature not reported" : `${row.Temp} °C`;
  const state = row.AgS === "Virgin" ? "virgin" : row.ageing_detail;
  if (row.source_id === "Zhou_2022_WaterRes") return `PE ${row.series_id.replace(/.*_/, "")} | virgin`;
  return `${row.ReT} | ${state} | ${temp}`;
}

const rows = parseCsv(await fs.readFile(appendPath, "utf8"));
if (rows.length !== 106) throw new Error(`Expected 106 rows, found ${rows.length}`);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const sheet = workbook.worksheets.getItem("qe_model_cu");
sheet.getRange("R869:U974").values = rows.map((row) => [
  row.series_id,
  label(row),
  row.ageing_type || null,
  row.ageing_detail || null,
]);
workbook.recalculate();
const check = await workbook.inspect({
  kind: "table",
  sheetId: "qe_model_cu",
  range: "P869:U974",
  include: "values,formulas",
  tableMaxRows: 106,
  tableMaxCols: 6,
  maxChars: 18000,
});
console.log(check.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan after Priority-1 series metadata fill",
});
console.log(errors.ndjson);
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(JSON.stringify({ status: "saved", rowsFilled: rows.length }));
