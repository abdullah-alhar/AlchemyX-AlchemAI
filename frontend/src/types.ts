export type Source = {
  source_family_id: string;
  source_number: number;
  title: string;
  document_type: string;
  formats: string[];
  supports: string[];
  role: string;
  authority_assessment: string;
  authority_summary?: string;
  reason?: string;
  chunk_ids: string[];
  chunks: number;
  chunk_numbers: number[];
  excerpt: string;
  entity_match: string;
  visibility: "evidence" | "context" | "excluded";
};

export type TimelineEntry = {
  iteration: number;
  query: string;
  retrieved: number;
  reranked: number;
  distinct_sources: number;
  new_evidence: number;
  conflict_status: string;
  candidate_values: string[];
  missing: string[];
  next_queries: string[];
  sufficient: boolean;
  sufficiency_label: string;
  stop_reason_label: string;
  elapsed_seconds: number | null;
  retrieval_seconds: number | null;
  rerank_seconds: number | null;
  sufficiency_seconds: number | null;
};

export type ResearchResult = {
  question: string;
  effective_question: string;
  used_conversation_context: boolean;
  clean_answer: string;
  iterations: number;
  stop_reason_label: string;
  research_run_id: string;
  sufficient: boolean;
  sources_retrieved: number;
  chunks_retrieved: number;
  conflict: {
    detected: boolean;
    resolved: boolean;
    state: string;
    status: string;
    candidate_values: string[];
    resolution_summary: string;
    selected_value: string;
  };
  sources: Source[];
  related_sources: Source[];
  excluded_sources: Source[];
  timeline: TimelineEntry[];
  timings: Record<string, number | null>;
};

export type ArchiveResult = {
  id: string;
  source_doc: string;
  chunk_index: number;
  text: string;
  score: number;
};
