import type { ArchiveResult, ResearchResult } from "./types";

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

const sourceBase = {
  document_type: "PDF",
  formats: ["PDF"] as string[],
  role: "supporting",
  authority_assessment: "Strong",
  chunk_ids: ["ashen_codex_chunk4"] as string[],
  chunks: 1,
  chunk_numbers: [4] as number[],
  entity_match: "Exact",
} as const;

export async function askResearch(question: string, history: ResearchResult[]): Promise<ResearchResult> {
  await wait(700);
  const trimmed = question.trim();
  const isFollowUp = /^(what about|how about|cite that|explain that|why is that)/i.test(trimmed);
  const previous = history.length > 0 ? history[history.length - 1] : undefined;

  return {
    question: trimmed,
    effective_question:
      isFollowUp && previous
        ? `Previous question: ${previous.question}\nPrevious answer: ${previous.clean_answer.slice(0, 220)}\nFollow-up question: ${trimmed}`
        : trimmed,
    used_conversation_context: Boolean(isFollowUp && previous),
    clean_answer:
      "The archive points to a controlled recovery process rather than a spontaneous event. The strongest evidence describes field notes being cross-checked against the recovered codex, then narrowed through a second retrieval pass before the final claim was accepted [1, 2].",
    iterations: 2,
    stop_reason_label: "Sufficient evidence",
    research_run_id: Math.random().toString(16).slice(2, 8),
    sufficient: true,
    sources_retrieved: 2,
    chunks_retrieved: 5,
    conflict: {
      detected: true,
      resolved: true,
      state: "resolved",
      status: "Resolved",
      candidate_values: ["controlled recovery", "unverified discovery"],
      resolution_summary: "The field log and codex agree; the weaker claim appears only in an unsourced summary.",
      selected_value: "controlled recovery",
    },
    sources: [
      {
        ...sourceBase,
        source_family_id: "ashen_codex",
        source_number: 1,
        title: "Ashen Codex Field Notes",
        supports: ["Supports: field notes were cross-checked before acceptance"],
        authority_summary: "Primary field evidence with matching archive references.",
        excerpt:
          "The recovered folio was not accepted until it was compared with the field register and the earlier witness transcript.",
        visibility: "evidence",
      },
      {
        ...sourceBase,
        source_family_id: "expedition_register",
        source_number: 2,
        title: "Expedition Register II",
        supports: ["Supports: second-pass verification confirmed the claim"],
        chunk_ids: ["expedition_register_ii_chunk11"],
        chunk_numbers: [11],
        excerpt:
          "A second review marked the first report as incomplete and retained only the entries with direct artifact correspondence.",
        visibility: "evidence",
      },
    ],
    related_sources: [
      {
        ...sourceBase,
        source_family_id: "archive_summary",
        source_number: 0,
        title: "Archive Summary",
        role: "supporting",
        authority_assessment: "Available evidence",
        supports: ["Provides related context"],
        chunk_ids: ["archive_summary_chunk2"],
        chunk_numbers: [2],
        excerpt: "Later summaries repeat the discovery story but do not include the full source trail.",
        visibility: "context",
      },
    ],
    excluded_sources: [
      {
        ...sourceBase,
        source_family_id: "oral_fragment",
        source_number: 0,
        title: "Oral Fragment",
        role: "supporting",
        authority_assessment: "Lower or unresolved",
        supports: ["Retrieved but excluded from direct evidence"],
        chunk_ids: ["oral_fragment_chunk7"],
        chunk_numbers: [7],
        excerpt: "A conflicting version is mentioned without dates or supporting archive references.",
        entity_match: "Related",
        visibility: "excluded",
      },
    ],
    timeline: [
      {
        iteration: 1,
        query: trimmed,
        retrieved: 8,
        reranked: 5,
        distinct_sources: 4,
        new_evidence: 3,
        conflict_status: "Detected",
        candidate_values: ["controlled recovery", "unverified discovery"],
        missing: ["Direct confirmation from the register"],
        next_queries: ["expedition register verification recovery process"],
        sufficient: false,
        sufficiency_label: "No",
        stop_reason_label: "",
        elapsed_seconds: 4.8,
        retrieval_seconds: 1.3,
        rerank_seconds: 0.9,
        sufficiency_seconds: 2.1,
      },
      {
        iteration: 2,
        query: "expedition register verification recovery process",
        retrieved: 6,
        reranked: 5,
        distinct_sources: 3,
        new_evidence: 2,
        conflict_status: "Resolved",
        candidate_values: ["controlled recovery"],
        missing: [],
        next_queries: [],
        sufficient: true,
        sufficiency_label: "Yes",
        stop_reason_label: "Sufficient evidence",
        elapsed_seconds: 5.4,
        retrieval_seconds: 1.1,
        rerank_seconds: 0.8,
        sufficiency_seconds: 1.9,
      },
    ],
    timings: {
      agent_loop: 10.2,
      retrieval: 2.4,
      reranking: 1.7,
      sufficiency: 4.0,
      final_generation: 2.6,
      end_to_end: 12.8,
    },
  };
}

export async function searchArchive(query: string): Promise<ArchiveResult[]> {
  await wait(350);
  return [
    {
      id: "ashen_codex_chunk4",
      source_doc: "ashen_codex.scan.pdf",
      chunk_index: 4,
      text: `Top archive match for "${query}". This chunk contains the strongest direct evidence and should become a source card in the final answer.`,
      score: 0.91,
    },
    {
      id: "expedition_register_ii_chunk11",
      source_doc: "expedition_register_ii.pdf",
      chunk_index: 11,
      text: "A supporting register entry that narrows the claim and confirms the selected interpretation.",
      score: 0.84,
    },
    {
      id: "archive_summary_chunk2",
      source_doc: "archive_summary.md",
      chunk_index: 2,
      text: "Useful contextual material, but weaker than the primary source trail.",
      score: 0.72,
    },
  ];
}
