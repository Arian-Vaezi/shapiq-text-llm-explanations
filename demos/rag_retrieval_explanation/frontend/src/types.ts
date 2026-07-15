export type Settings = {
  value_modes: string[];
  interaction_index: "SV" | "k-SII" | "STII" | "FSII";
  max_order: number;
  budget: number;
  model_path: string;
  model_id: string | null;
  n_ctx: number;
  n_gpu_layers: number;
  n_threads: number;
  max_new_tokens: number;
};

export type RetrievalSettings = {
  method: "Dense embeddings" | "TF-IDF";
  topK: number;
  chunkWords: number;
  chunkOverlap: number;
  embeddingModelId: string;
  embeddingDevice: string;
};

export type ModelEntry = {
  id: string;
  display_name: string;
  family: string;
  params: string;
  filename: string;
  description: string;
  path: string;
  available: boolean;
  repo_id: string;
  download_url: string;
  download_command: string;
};

export type Meta = {
  scenarios: { name: string; question: string; takeaway: string }[];
  value_functions: {
    name: string;
    description: string;
    formula: string;
    requires_model: boolean;
  }[];
  defaults: Settings & {
    retrieval_method: string;
    retrieval_top_k: number;
    chunk_words: number;
    chunk_overlap: number;
    embedding_model_id: string;
    embedding_device: string;
    model_id: string;
  };
  models: ModelEntry[];
  default_model_id: string;
};

export type Attribution = {
  player: number;
  chunk: string;
  attribution: number;
  loo: number | null;
};

export type CurvePoint = { step: number; score: number; label: string };

export type ExplanationResult = {
  value_name: string;
  value_description: string;
  value_formula: string;
  full_score: number;
  empty_score: number;
  top_chunk: string;
  top_score: number;
  attributions: Attribution[];
  pairwise_matrix: number[][];
  strongest_pair: string;
  strongest_pair_value: number;
  faithfulness: {
    full_score: number;
    score_without_top: number;
    score_with_top_only: number;
    deletion_drop: number;
    sufficiency_gap: number;
  };
  deletion_curve: CurvePoint[];
  insertion_curve: CurvePoint[];
  random_deletion_baseline: number[];
  random_insertion_baseline: number[];
  retrieval_deletion_baseline: number[] | null;
};

export type AnswerTargetResult = {
  target_type: "reference_answer" | "generated_answer";
  label: string;
  answer: string;
  results: ExplanationResult[];
};

export type RunResult = {
  run_name: string;
  source: string;
  answer_target_type: "reference_answer" | "generated_answer";
  retrieval_method: string;
  retrieval_top_k: number;
  question: string;
  answer: string;
  takeaway: string;
  chunks: {
    title: string;
    text: string;
    page_number: number | null;
    chunk_type: string;
    section_title: string;
    retrieval_score: number | null;
    dense_score: number | null;
    keyword_score: number | null;
    rerank_score: number | null;
    flags: string[];
  }[];
  retrieval_debug: Record<string, unknown> | null;
  results: ExplanationResult[];
  targets: AnswerTargetResult[];
};

export type ChatMessage = {
  id: string;
  question: string;
  result: RunResult;
  resultIndex: number;
  modelDisplayName: string | null;
  interactionIndex: string;
};

export type AppMode = "single" | "compare";
export type AppView = "chat" | "compare" | "detail";

export type ModelRunResult = {
  model_id: string;
  model_display_name: string;
  status: "completed" | "missing_model" | "failed" | "running";
  error?: string;
  expected_path?: string;
  result?: RunResult;
};

export type CompareRun = {
  id: string;
  question: string;
  source: string;
  model_results: ModelRunResult[];
};

export type ComparisonResponse = {
  question: string;
  source: string;
  results: ModelRunResult[];
};
