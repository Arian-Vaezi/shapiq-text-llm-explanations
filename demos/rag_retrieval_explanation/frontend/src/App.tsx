import { useEffect, useState } from "react";
import {
  getMeta,
  runComparePdf,
  runCompareSample,
  runPdf,
  runSample,
} from "./api";
import type {
  AppMode,
  AppView,
  ChatMessage,
  CompareRun,
  Meta,
  RetrievalSettings,
  RunResult,
  Settings,
} from "./types";
import { Sidebar } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";
import { CompareView } from "./components/CompareView";
import { EvidenceDetailView } from "./components/EvidenceDetailView";

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [mode, setMode] = useState<AppMode>("single");
  const [source, setSource] = useState<"sample" | "pdf">("sample");
  const [scenario, setScenario] = useState("");
  const [settings, setSettings] = useState<Settings | null>(null);
  const [retrieval, setRetrieval] = useState<RetrievalSettings | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [referenceAnswer, setReferenceAnswer] = useState("");

  // Single-mode chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  // Compare-mode state
  const [compareModelIds, setCompareModelIds] = useState<string[]>([]);
  const [compareRun, setCompareRun] = useState<CompareRun | null>(null);

  // View routing
  const [view, setView] = useState<AppView>("chat");
  const [detailFrom, setDetailFrom] = useState<"chat" | "compare">("chat");
  const [detailMessageId, setDetailMessageId] = useState<string | null>(null);
  const [detailCompareModelId, setDetailCompareModelId] = useState<string | null>(null);
  const [detailResultIndex, setDetailResultIndex] = useState(0);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getMeta()
      .then((data) => {
        setMeta(data);
        setScenario(data.scenarios[0]?.name ?? "");
        setSettings({
          ...data.defaults,
          model_id: data.default_model_id ?? null,
        });
        setRetrieval({
          method: data.defaults.retrieval_method as RetrievalSettings["method"],
          topK: data.defaults.retrieval_top_k,
          chunkWords: data.defaults.chunk_words,
          chunkOverlap: data.defaults.chunk_overlap,
          embeddingModelId: data.defaults.embedding_model_id,
          embeddingDevice: data.defaults.embedding_device,
        });
        setCompareModelIds(
          data.default_model_id ? [data.default_model_id] : []
        );
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  function handleModeChange(newMode: AppMode) {
    setMode(newMode);
    setView(newMode === "compare" ? "compare" : "chat");
    setError("");
  }

  async function execute() {
    if (!settings || !meta || !retrieval) return;
    setBusy(true);
    setError("");

    try {
      if (mode === "single") {
        const next: RunResult =
          source === "sample"
            ? await runSample(scenario, settings)
            : await runPdf(file!, question, referenceAnswer, settings, retrieval);
        const selectedModel = meta.models?.find((m) => m.id === settings.model_id);
        const msg: ChatMessage = {
          id: crypto.randomUUID(),
          question: next.question,
          result: next,
          resultIndex: 0,
          modelDisplayName: selectedModel?.display_name ?? null,
          interactionIndex: settings.interaction_index,
        };
        setMessages((prev) => [...prev, msg]);
        if (source === "pdf") setQuestion("");
      } else {
        const response =
          source === "sample"
            ? await runCompareSample(scenario, settings, compareModelIds)
            : await runComparePdf(
                file!,
                question,
                referenceAnswer,
                settings,
                compareModelIds,
                retrieval,
              );
        const run: CompareRun = {
          id: crypto.randomUUID(),
          question: response.question,
          source: response.source,
          model_results: response.results,
        };
        setCompareRun(run);
        if (source === "pdf") setQuestion("");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Run failed");
    } finally {
      setBusy(false);
    }
  }

  function openChatDetail(msgId: string) {
    setDetailMessageId(msgId);
    setDetailFrom("chat");
    setDetailResultIndex(0);
    setView("detail");
  }

  function openCompareDetail(modelId: string) {
    setDetailCompareModelId(modelId);
    setDetailFrom("compare");
    setDetailResultIndex(0);
    setView("detail");
  }

  function closeDetail() {
    setView(detailFrom === "chat" ? "chat" : "compare");
  }

  if (!meta || !settings || !retrieval) {
    return <div className="loading-state">Loading workspace…</div>;
  }

  const selectedScenario = meta.scenarios.find((s) => s.name === scenario);
  const canRun =
    mode === "single"
      ? source === "sample"
        ? Boolean(scenario)
        : Boolean(file && question.trim())
      : compareModelIds.length > 0 &&
        (source === "sample"
          ? Boolean(scenario)
          : Boolean(file && question.trim()));

  const detailMsg = detailMessageId
    ? messages.find((m) => m.id === detailMessageId) ?? null
    : null;
  const detailModelResult =
    detailCompareModelId && compareRun
      ? compareRun.model_results.find((r) => r.model_id === detailCompareModelId) ?? null
      : null;

  return (
    <div className="app-shell">
      <header className="masthead">
        <span className="app-title">RAG Shapley</span>
      </header>

      <div className="workspace">
        <Sidebar
          meta={meta}
          settings={settings}
          onSettingsChange={setSettings}
          retrieval={retrieval}
          onRetrievalChange={setRetrieval}
          mode={mode}
          onModeChange={handleModeChange}
          source={source}
          onSourceChange={setSource}
          scenario={scenario}
          onScenarioChange={setScenario}
          file={file}
          onFileChange={setFile}
          referenceAnswer={referenceAnswer}
          onReferenceAnswerChange={setReferenceAnswer}
          compareModelIds={compareModelIds}
          onCompareModelIdsChange={setCompareModelIds}
        />

        <main className="main-column">
          {view === "chat" && (
            <ChatView
              messages={messages}
              source={source}
              selectedScenario={selectedScenario}
              question={question}
              onQuestionChange={setQuestion}
              canRun={canRun}
              busy={busy}
              error={error}
              onRun={execute}
              onViewDetail={openChatDetail}
            />
          )}

          {view === "compare" && (
            <CompareView
              compareRun={compareRun}
              source={source}
              selectedScenario={selectedScenario}
              compareModelIds={compareModelIds}
              question={question}
              onQuestionChange={setQuestion}
              canRun={canRun}
              busy={busy}
              error={error}
              onRun={execute}
              onViewDetail={openCompareDetail}
            />
          )}

          {view === "detail" && (
            <>
              {detailFrom === "chat" && detailMsg ? (
                <EvidenceDetailView
                  question={detailMsg.question}
                  result={detailMsg.result}
                  modelDisplayName={detailMsg.modelDisplayName ?? undefined}
                  resultIndex={detailResultIndex}
                  onResultIndexChange={setDetailResultIndex}
                  backLabel="Back to chat"
                  onBack={closeDetail}
                />
              ) : detailFrom === "compare" &&
                detailModelResult?.result &&
                compareRun ? (
                <EvidenceDetailView
                  question={compareRun.question}
                  result={detailModelResult.result}
                  modelDisplayName={detailModelResult.model_display_name}
                  resultIndex={detailResultIndex}
                  onResultIndexChange={setDetailResultIndex}
                  backLabel="Back to comparison"
                  onBack={closeDetail}
                />
              ) : null}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
