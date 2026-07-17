import { useState } from "react";
import { ChevronDown, ChevronRight, Upload } from "lucide-react";
import type {
  AppMode,
  Meta,
  ModelEntry,
  RetrievalSettings,
  Settings,
} from "../types";

interface Props {
  meta: Meta;
  settings: Settings;
  onSettingsChange: (s: Settings) => void;
  retrieval: RetrievalSettings;
  onRetrievalChange: (s: RetrievalSettings) => void;
  mode: AppMode;
  onModeChange: (m: AppMode) => void;
  source: "sample" | "pdf";
  onSourceChange: (s: "sample" | "pdf") => void;
  scenario: string;
  onScenarioChange: (s: string) => void;
  file: File | null;
  onFileChange: (f: File | null) => void;
  referenceAnswer: string;
  onReferenceAnswerChange: (answer: string) => void;
  compareModelIds: string[];
  onCompareModelIdsChange: (ids: string[]) => void;
}

const INDEX_HINTS: Record<string, string> = {
  SV: "Individual marginal contributions only.",
  "k-SII": "Captures pairwise and higher-order synergies between chunks.",
  STII: "Permutation-sampling Taylor Interaction Index.",
  FSII: "Regression-based, faithful to Shapley axioms.",
};

export function Sidebar({
  meta,
  settings,
  onSettingsChange,
  retrieval,
  onRetrievalChange,
  mode,
  onModeChange,
  source,
  onSourceChange,
  scenario,
  onScenarioChange,
  file,
  onFileChange,
  referenceAnswer,
  onReferenceAnswerChange,
  compareModelIds,
  onCompareModelIdsChange,
}: Props) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  function set<K extends keyof Settings>(key: K, val: Settings[K]) {
    onSettingsChange({ ...settings, [key]: val });
  }

  function setRetrieval<K extends keyof RetrievalSettings>(
    key: K,
    val: RetrievalSettings[K],
  ) {
    onRetrievalChange({ ...retrieval, [key]: val });
  }

  const selectedScenario = meta.scenarios.find((s) => s.name === scenario);
  const activeVf = meta.value_functions.find(
    (f) => f.name === settings.value_modes[0]
  );
  const selectedModel: ModelEntry | undefined = meta.models?.find(
    (m) => m.id === settings.model_id
  );
  const modelNeedsWarning =
    (activeVf?.requires_model ?? false) &&
    selectedModel !== undefined &&
    !selectedModel.available;

  function toggleCompareModel(id: string, checked: boolean) {
    if (checked) {
      onCompareModelIdsChange([...compareModelIds, id]);
    } else {
      onCompareModelIdsChange(compareModelIds.filter((x) => x !== id));
    }
  }

  return (
    <aside className="sidebar">
      {/* Mode toggle */}
      <div className="mode-toggle">
        <button
          className={mode === "single" ? "active" : ""}
          onClick={() => onModeChange("single")}
        >
          Single
        </button>
        <button
          className={mode === "compare" ? "active" : ""}
          onClick={() => onModeChange("compare")}
        >
          Compare
        </button>
      </div>

      {/* Source toggle */}
      <div className="source-toggle">
        <button
          className={source === "sample" ? "active" : ""}
          onClick={() => onSourceChange("sample")}
        >
          Samples
        </button>
        <button
          className={source === "pdf" ? "active" : ""}
          onClick={() => onSourceChange("pdf")}
        >
          PDF
        </button>
      </div>

      {/* Source-specific inputs */}
      {source === "sample" ? (
        <div className="sidebar-fields">
          {selectedScenario?.takeaway && (
            <div className="purpose-block">
              <span className="form-label">Purpose</span>
              <p className="purpose-text">{selectedScenario.takeaway}</p>
            </div>
          )}
          <div className="form-field">
            <label className="form-label" htmlFor="scenario-select">
              Scenario
            </label>
            <select
              id="scenario-select"
              value={scenario}
              onChange={(e) => onScenarioChange(e.target.value)}
            >
              {meta.scenarios.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name}
                </option>
              ))}
            </select>
          </div>
          {selectedScenario && (
            <div className="form-field">
              <span className="form-label">Question</span>
              <p className="scenario-question">{selectedScenario.question}</p>
            </div>
          )}
        </div>
      ) : (
        <div className="sidebar-fields">
          <div className="form-field">
            <label className="upload-zone">
              <Upload size={15} />
              <span className="upload-name">
                {file ? file.name : "Choose a PDF"}
              </span>
              <span className="upload-hint">
                Selectable text, processed locally
              </span>
              <input
                type="file"
                accept=".pdf,application/pdf"
                onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
              />
            </label>
          </div>
          <label className="field-label">
            Reference answer
            <textarea
              value={referenceAnswer}
              onChange={(e) => onReferenceAnswerChange(e.target.value)}
              placeholder="Optional: enter a human-verified answer."
              rows={3}
            />
            <span className="field-hint">
              Adds a separate gold/reference support explanation beside the generated answer.
            </span>
          </label>
          <div className="retrieval-settings">
            <span className="form-label">Retrieval</span>
            <label className="field-label">
              Retriever
              <select
                value={retrieval.method}
                onChange={(e) =>
                  setRetrieval(
                    "method",
                    e.target.value as RetrievalSettings["method"],
                  )
                }
              >
                <option value="Dense embeddings">BGE-base embeddings</option>
                <option value="TF-IDF">TF-IDF lexical baseline</option>
              </select>
              <span className="field-hint">
                {retrieval.method === "Dense embeddings"
                  ? "Primary setting from the controlled evaluation."
                  : "Lexical retrieval baseline for comparison."}
              </span>
            </label>
            <div className="field-row">
              <label className="field-label">
                Top-k
                <input
                  type="number"
                  min={2}
                  max={8}
                  value={retrieval.topK}
                  onChange={(e) => setRetrieval("topK", Number(e.target.value))}
                />
              </label>
              <label className="field-label">
                Chunk words
                <input
                  type="number"
                  min={80}
                  max={600}
                  value={retrieval.chunkWords}
                  onChange={(e) =>
                    setRetrieval("chunkWords", Number(e.target.value))
                  }
                />
              </label>
            </div>
          </div>
        </div>
      )}

      {source === "sample" && (
        <div className="controlled-note">
          <strong>Controlled coalition demo</strong>
          <span>
            These sample chunks are predefined to demonstrate Shapley behavior.
            They are not evidence of retrieval accuracy.
          </span>
        </div>
      )}

      {/* Attribution settings */}
      <div className="sidebar-settings">
        <label className="field-label">
          Value function
          <select
            value={settings.value_modes[0]}
            onChange={(e) => set("value_modes", [e.target.value])}
          >
            {meta.value_functions.map((f) => (
              <option key={f.name} value={f.name}>
                {f.name}
              </option>
            ))}
          </select>
          {activeVf?.description && (
            <span className="field-hint">{activeVf.description}</span>
          )}
        </label>

        <label className="field-label">
          Interaction index
          <select
            value={settings.interaction_index}
            onChange={(e) =>
              set(
                "interaction_index",
                e.target.value as Settings["interaction_index"]
              )
            }
          >
            <option value="SV">SV — Shapley values</option>
            <option value="k-SII">k-SII — Shapley Interaction Index</option>
            <option value="STII">STII — Taylor Interaction Index</option>
            <option value="FSII">FSII — Faithful Shapley Interaction Index</option>
          </select>
          <span className="field-hint">
            {INDEX_HINTS[settings.interaction_index]}
          </span>
        </label>

        {/* Single mode: single model selector */}
        {mode === "single" && meta.models && meta.models.length > 0 && (
          <label className="field-label">
            Model
            <select
              value={settings.model_id ?? ""}
              onChange={(e) => set("model_id", e.target.value || null)}
            >
              {meta.models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name}
                  {!m.available ? " (not found)" : ""}
                </option>
              ))}
            </select>
            {selectedModel && (
              <span
                className={`field-hint${modelNeedsWarning ? " warning" : ""}`}
              >
                {selectedModel.available
                  ? `Available · ${selectedModel.filename}`
                  : `Not found: ${selectedModel.filename}`}
              </span>
            )}
            {selectedModel && !selectedModel.available && (
              <span className="field-hint model-download-hint">
                Download:{" "}
                <code className="model-download-cmd">
                  {selectedModel.download_command}
                </code>
              </span>
            )}
          </label>
        )}

        {/* Compare mode: multi-checkbox model list */}
        {mode === "compare" && meta.models && meta.models.length > 0 && (
          <div className="field-label">
            Models
            <span className="model-checkbox-count">
              ({compareModelIds.length} selected)
            </span>
            <div className="model-checkbox-list">
              {meta.models.map((m) => (
                <label
                  key={m.id}
                  className={`model-checkbox-item${!m.available ? " unavailable" : ""}`}
                  title={!m.available ? m.download_command : undefined}
                >
                  <input
                    type="checkbox"
                    checked={compareModelIds.includes(m.id)}
                    onChange={(e) => toggleCompareModel(m.id, e.target.checked)}
                  />
                  <span className="model-checkbox-name">{m.display_name}</span>
                  <span
                    className={`model-checkbox-avail ${m.available ? "ok" : "miss"}`}
                  >
                    {m.available ? "✓" : "✗"}
                  </span>
                </label>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Advanced */}
      <button
        className="advanced-toggle"
        onClick={() => setAdvancedOpen(!advancedOpen)}
      >
        {advancedOpen ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
        Advanced
      </button>

      {advancedOpen && (
        <div className="sidebar-advanced">
          <div className="field-row">
            <label className="field-label">
              Budget
              <input
                type="number"
                min={16}
                max={512}
                value={settings.budget}
                onChange={(e) => set("budget", Number(e.target.value))}
              />
            </label>
            <label className="field-label">
              Max order
              <input
                type="number"
                min={1}
                max={3}
                value={settings.max_order}
                onChange={(e) => set("max_order", Number(e.target.value))}
              />
            </label>
          </div>
          <label className="field-label">
            Context tokens
            <input
              type="number"
              value={settings.n_ctx}
              onChange={(e) => set("n_ctx", Number(e.target.value))}
            />
          </label>
          <div className="field-row">
            <label className="field-label">
              GPU layers
              <input
                type="number"
                value={settings.n_gpu_layers}
                onChange={(e) => set("n_gpu_layers", Number(e.target.value))}
              />
            </label>
            <label className="field-label">
              CPU threads
              <input
                type="number"
                value={settings.n_threads}
                onChange={(e) => set("n_threads", Number(e.target.value))}
              />
            </label>
          </div>
          <label className="field-label">
            Max new tokens
            <input
              type="number"
              min={16}
              max={512}
              value={settings.max_new_tokens}
              onChange={(e) => set("max_new_tokens", Number(e.target.value))}
            />
          </label>
        </div>
      )}
    </aside>
  );
}
