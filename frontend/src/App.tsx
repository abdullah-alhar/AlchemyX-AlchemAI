import { FormEvent, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Clock3,
  FileSearch,
  Layers3,
  MessageSquareText,
  Search,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import { askResearch, searchArchive } from "./mockApi";
import type { ArchiveResult, ResearchResult, Source, TimelineEntry } from "./types";

const sampleQuestions = [
  "What does the archive say about the recovered codex?",
  "Was the discovery verified by a second source?",
  "What about the conflicting oral fragment?",
];

export function App() {
  const [activeView, setActiveView] = useState<"research" | "archive">("research");
  const [question, setQuestion] = useState(sampleQuestions[0]);
  const [archiveQuery, setArchiveQuery] = useState("recovered codex verification");
  const [history, setHistory] = useState<ResearchResult[]>([]);
  const [activeResult, setActiveResult] = useState<ResearchResult | null>(null);
  const [archiveResults, setArchiveResults] = useState<ArchiveResult[]>([]);
  const [isResearching, setIsResearching] = useState(false);
  const [isSearching, setIsSearching] = useState(false);

  const selectedMetrics = useMemo(() => {
    if (!activeResult) {
      return [
        ["Status", "Ready"],
        ["Sources", "0"],
        ["Iterations", "0"],
        ["Runtime", "n/a"],
      ];
    }
    return [
      ["Status", activeResult.stop_reason_label],
      ["Sources", String(activeResult.sources_retrieved)],
      ["Iterations", String(activeResult.iterations)],
      ["Runtime", formatSeconds(activeResult.timings.end_to_end)],
    ];
  }, [activeResult]);

  async function submitQuestion(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || isResearching) return;
    setIsResearching(true);
    const result = await askResearch(question, history);
    setActiveResult(result);
    setHistory((items) => [result, ...items]);
    setIsResearching(false);
  }

  async function submitArchiveSearch(event: FormEvent) {
    event.preventDefault();
    if (!archiveQuery.trim() || isSearching) return;
    setIsSearching(true);
    setArchiveResults(await searchArchive(archiveQuery));
    setIsSearching(false);
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">A</div>
          <div>
            <strong>AlchemAI</strong>
            <span>Research Console</span>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Primary">
          <button className={activeView === "research" ? "active" : ""} onClick={() => setActiveView("research")}>
            <MessageSquareText size={18} />
            Research
          </button>
          <button className={activeView === "archive" ? "active" : ""} onClick={() => setActiveView("archive")}>
            <FileSearch size={18} />
            Archive Search
          </button>
        </nav>

        <section className="panel compact">
          <div className="panel-title">
            <Activity size={16} />
            Run Snapshot
          </div>
          <div className="metric-stack">
            {selectedMetrics.map(([label, value]) => (
              <div className="mini-metric" key={label}>
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>
        </section>

        <section className="panel compact history-panel">
          <div className="panel-title">
            <Clock3 size={16} />
            Session History
          </div>
          {history.length === 0 ? (
            <p className="muted">Research turns will appear here after the first mock run.</p>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <button key={item.research_run_id} onClick={() => setActiveResult(item)}>
                  <span>{item.question}</span>
                  <small>{item.stop_reason_label}</small>
                </button>
              ))}
            </div>
          )}
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Mock frontend contract</p>
            <h1>{activeView === "research" ? "Evidence-Grounded Research" : "Explore Indexed Evidence"}</h1>
          </div>
          <div className="status-chip">
            <ShieldCheck size={16} />
            FastAPI-ready mock mode
          </div>
        </header>

        {activeView === "research" ? (
          <ResearchView
            question={question}
            setQuestion={setQuestion}
            submitQuestion={submitQuestion}
            isResearching={isResearching}
            activeResult={activeResult}
          />
        ) : (
          <ArchiveView
            query={archiveQuery}
            setQuery={setArchiveQuery}
            submitSearch={submitArchiveSearch}
            isSearching={isSearching}
            results={archiveResults}
          />
        )}
      </section>
    </main>
  );
}

function ResearchView({
  question,
  setQuestion,
  submitQuestion,
  isResearching,
  activeResult,
}: {
  question: string;
  setQuestion: (value: string) => void;
  submitQuestion: (event: FormEvent) => void;
  isResearching: boolean;
  activeResult: ResearchResult | null;
}) {
  return (
    <div className="research-grid">
      <section className="primary-column">
        <form className="query-box" onSubmit={submitQuestion}>
          <label htmlFor="question">Ask the archive</label>
          <div className="query-row">
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={3}
            />
            <button className="run-button" type="submit" disabled={isResearching}>
              {isResearching ? <Sparkles size={18} /> : <ArrowRight size={18} />}
              {isResearching ? "Running" : "Run"}
            </button>
          </div>
        </form>

        {activeResult ? (
          <>
            <AnswerPanel result={activeResult} />
            <TimelinePanel timeline={activeResult.timeline} />
          </>
        ) : (
          <EmptyState />
        )}
      </section>

      <aside className="detail-column">
        {activeResult ? (
          <>
            <ConflictPanel result={activeResult} />
            <SourcesPanel title="Evidence Sources" sources={activeResult.sources} />
            <SourcesPanel title="Related Sources" sources={activeResult.related_sources} />
            <SourcesPanel title="Excluded Evidence" sources={activeResult.excluded_sources} />
            <DiagnosticsPanel result={activeResult} />
          </>
        ) : (
          <section className="panel">
            <div className="panel-title">
              <Layers3 size={16} />
              Interface Contract
            </div>
            <p className="muted">
              This screen is using mock data shaped like the backend adapter response. The API swap later should only
              change the data client.
            </p>
          </section>
        )}
      </aside>
    </div>
  );
}

function ArchiveView({
  query,
  setQuery,
  submitSearch,
  isSearching,
  results,
}: {
  query: string;
  setQuery: (value: string) => void;
  submitSearch: (event: FormEvent) => void;
  isSearching: boolean;
  results: ArchiveResult[];
}) {
  return (
    <section className="archive-layout">
      <form className="query-box search-box" onSubmit={submitSearch}>
        <label htmlFor="archive-query">Search indexed chunks</label>
        <div className="single-line-query">
          <input id="archive-query" value={query} onChange={(event) => setQuery(event.target.value)} />
          <button className="icon-button" type="submit" disabled={isSearching} aria-label="Search archive">
            <Search size={18} />
          </button>
        </div>
      </form>

      <div className="archive-results">
        {results.length === 0 ? (
          <EmptyState compact />
        ) : (
          results.map((result) => (
            <article className="result-row" key={result.id}>
              <div>
                <strong>{readableDoc(result.source_doc)}</strong>
                <span>Chunk {result.chunk_index}</span>
              </div>
              <p>{result.text}</p>
              <small>Score {result.score.toFixed(2)}</small>
            </article>
          ))
        )}
      </div>
    </section>
  );
}

function AnswerPanel({ result }: { result: ResearchResult }) {
  return (
    <section className="panel answer-panel">
      <div className="panel-title">
        <BookOpen size={16} />
        Answer
        <span className="run-id">Run {result.research_run_id}</span>
      </div>
      {result.used_conversation_context && <p className="context-note">Used previous turn context.</p>}
      <p className="answer-text">{result.clean_answer}</p>
      <div className="answer-footer">
        <span>{result.sources_retrieved} evidence sources</span>
        <span>{result.chunks_retrieved} chunks retrieved</span>
        <span>{result.iterations} iterations</span>
      </div>
    </section>
  );
}

function TimelinePanel({ timeline }: { timeline: TimelineEntry[] }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <Activity size={16} />
        Research Timeline
      </div>
      <div className="timeline">
        {timeline.map((entry) => (
          <article className="timeline-item" key={entry.iteration}>
            <div className="timeline-marker">{entry.iteration}</div>
            <div className="timeline-body">
              <div className="timeline-head">
                <strong>{entry.query}</strong>
                <span>{entry.stop_reason_label || entry.conflict_status}</span>
              </div>
              <div className="timeline-metrics">
                <span>{entry.retrieved} retrieved</span>
                <span>{entry.reranked} reranked</span>
                <span>{entry.distinct_sources} sources</span>
                <span>{formatSeconds(entry.elapsed_seconds)}</span>
              </div>
              {entry.missing.length > 0 && <p>Missing: {entry.missing.join("; ")}</p>}
              {entry.next_queries.length > 0 && <p>Next: {entry.next_queries.join("; ")}</p>}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function SourcesPanel({ title, sources }: { title: string; sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <section className="panel">
      <div className="panel-title">
        <FileSearch size={16} />
        {title}
      </div>
      <div className="source-list">
        {sources.map((source) => (
          <article className={`source-card ${source.visibility}`} key={`${title}-${source.source_family_id}`}>
            <div className="source-head">
              <strong>{source.source_number ? `[${source.source_number}] ${source.title}` : source.title}</strong>
              <span>{source.formats.join(", ") || source.document_type}</span>
            </div>
            {source.supports.map((support) => (
              <p key={support}>{support}</p>
            ))}
            <small>{source.excerpt}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

function ConflictPanel({ result }: { result: ResearchResult }) {
  return (
    <section className="panel">
      <div className="panel-title">
        {result.conflict.detected ? <TriangleAlert size={16} /> : <CheckCircle2 size={16} />}
        Conflict Check
      </div>
      <div className="conflict-state">
        <strong>{result.conflict.status}</strong>
        <span>{result.sufficient ? "Sufficient" : "Needs review"}</span>
      </div>
      {result.conflict.resolution_summary && <p className="muted">{result.conflict.resolution_summary}</p>}
    </section>
  );
}

function DiagnosticsPanel({ result }: { result: ResearchResult }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <Clock3 size={16} />
        Diagnostics
      </div>
      <div className="diagnostic-grid">
        {Object.entries(result.timings).map(([label, value]) => (
          <div key={label}>
            <span>{label.replace(/_/g, " ")}</span>
            <strong>{formatSeconds(value)}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function EmptyState({ compact = false }: { compact?: boolean }) {
  return (
    <section className={`empty-state ${compact ? "compact-empty" : ""}`}>
      <Sparkles size={22} />
      <strong>Ready for a mock research run</strong>
      <span>Use the controls above to see the interface populated with adapter-shaped data.</span>
    </section>
  );
}

function formatSeconds(value: number | null | undefined) {
  if (value === null || value === undefined) return "n/a";
  return `${value.toFixed(1)}s`;
}

function readableDoc(value: string) {
  return value
    .replace(/\.[^.]+$/, "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}
