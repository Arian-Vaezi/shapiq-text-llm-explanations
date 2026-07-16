import type {
  ComparisonResponse,
  Meta,
  RetrievalSettings,
  RunResult,
  Settings,
} from "./types";

async function parse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Request failed");
  }
  return response.json() as Promise<T>;
}

export const getMeta = () => fetch("/api/meta").then(parse<Meta>);

export const runSample = (scenarioName: string, settings: Settings) =>
  fetch("/api/runs/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario_name: scenarioName, settings }),
  }).then(parse<RunResult>);

export function runPdf(
  file: File,
  question: string,
  referenceAnswer: string,
  settings: Settings,
  retrieval: RetrievalSettings,
) {
  const body = new FormData();
  body.append("file", file);
  body.append("question", question);
  body.append("reference_answer", referenceAnswer);
  body.append("settings_json", JSON.stringify(settings));
  body.append("retrieval_method", retrieval.method);
  body.append("retrieval_top_k", String(retrieval.topK));
  body.append("chunk_words", String(retrieval.chunkWords));
  body.append("chunk_overlap", String(retrieval.chunkOverlap));
  body.append("embedding_model_id", retrieval.embeddingModelId);
  body.append("embedding_device", retrieval.embeddingDevice);
  return fetch("/api/runs/pdf", { method: "POST", body }).then(parse<RunResult>);
}

export const runCompareSample = (
  scenarioName: string,
  settings: Settings,
  modelIds: string[],
) =>
  fetch("/api/runs/compare/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      scenario_name: scenarioName,
      settings,
      model_ids: modelIds,
    }),
  }).then(parse<ComparisonResponse>);

export function runComparePdf(
  file: File,
  question: string,
  referenceAnswer: string,
  settings: Settings,
  modelIds: string[],
  retrieval: RetrievalSettings,
) {
  const body = new FormData();
  body.append("file", file);
  body.append("question", question);
  body.append("reference_answer", referenceAnswer);
  body.append("settings_json", JSON.stringify(settings));
  body.append("model_ids_json", JSON.stringify(modelIds));
  body.append("retrieval_method", retrieval.method);
  body.append("retrieval_top_k", String(retrieval.topK));
  body.append("chunk_words", String(retrieval.chunkWords));
  body.append("chunk_overlap", String(retrieval.chunkOverlap));
  body.append("embedding_model_id", retrieval.embeddingModelId);
  body.append("embedding_device", retrieval.embeddingDevice);
  return fetch("/api/runs/compare/pdf", { method: "POST", body }).then(
    parse<ComparisonResponse>,
  );
}
