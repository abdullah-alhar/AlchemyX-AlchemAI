from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from .agent.agent_loop import AgentLoop
from .agent.evidence_adjudicator import EvidenceAdjudicator
from .agent.openrouter_client import OpenRouterClient
from .agent.sufficiency_checker import SufficiencyChecker
from .generation.answer_generator import AnswerGenerator
from .generation.citation_builder import build_citations
from .retrieval.main import create_agent_retrieval_pipeline
from .telemetry import log, reset_run_id, set_run_id


@dataclass
class AlchemyXSystem:
    agent: AgentLoop
    answer_generator: AnswerGenerator

    def ask(self, question, progress=None):
        research_run_id = uuid4().hex[:6]
        token = set_run_id(research_run_id)
        try:
            checker = getattr(self.agent, "sufficiency_checker", None)
            llm_client = getattr(checker, "llm_client", None)
            if llm_client is None:
                llm_client = getattr(self.answer_generator, "llm_client", None)
            if hasattr(llm_client, "call_log"):
                llm_client.call_log.clear()
            _notify(progress, "Starting the research loop.")
            # AgentLoop generates its own ID when called through older/custom
            # implementations; use the returned ID when it provides one.
            agent_result = self.agent.run(question, progress=progress)
            research_run_id = agent_result.get("research_run_id", research_run_id)
            reset_run_id(token)
            token = set_run_id(research_run_id)
            _notify(progress, "Generating the final answer from selected evidence.")
            started = perf_counter()
            generated_answer = self.answer_generator.generate(
                question=question,
                documents=agent_result["documents"],
                sufficiency_result=agent_result["result"],
            )
            final_generation_seconds = perf_counter() - started
            log(f"Final generation: {final_generation_seconds:.2f}s")
            _notify(progress, "Building source citations and run details.")
            citations = build_citations(agent_result["documents"])
            checker = getattr(self.agent, "sufficiency_checker", None)
            llm_client = getattr(checker, "llm_client", None)
            llm_calls = list(getattr(llm_client, "call_log", []) or [])
        finally:
            reset_run_id(token)

        timings = dict(agent_result.get("timings", {}))
        timings["final_generation"] = final_generation_seconds
        agent_loop_seconds = timings.get("agent_loop") or 0
        timings["end_to_end"] = agent_loop_seconds + final_generation_seconds

        return {
            "answer": (
                "The evidence does not support a definitive answer."
                if getattr(generated_answer, "last_parse_error", False)
                and "malformed output" in generated_answer.answer
                else generated_answer.answer
            ),
            "answer_evidence_ids": generated_answer.evidence_ids,
            "citations": citations,
            "documents": agent_result["documents"],
            "sufficiency_result": agent_result["result"],
            "iterations": agent_result["iterations"],
            "stop_reason": agent_result["stop_reason"],
            "research_run_id": research_run_id,
            "timeline": agent_result.get("timeline", []),
            "timings": timings,
            "internal_errors": _collect_internal_errors(agent_result, generated_answer),
            "llm_calls": llm_calls,
            "llm_call_counts": _llm_call_counts(llm_calls),
        }


def _collect_internal_errors(agent_result, generated_answer):
    errors = []
    result = agent_result.get("result")
    for owner in (result, getattr(result, "conflict_result", None)):
        for error in getattr(owner, "internal_errors", []) or []:
            if error not in errors:
                errors.append(error)
    return errors


def _llm_call_counts(calls):
    counts = {"sufficiency": 0, "adjudication": 0, "final_generation": 0, "repair_retries": 0}
    for call in calls:
        purpose = str(call.get("purpose", ""))
        if purpose.startswith("sufficiency"):
            counts["sufficiency"] += 1
        elif purpose.startswith("adjudication"):
            counts["adjudication"] += 1
        elif purpose == "final_generation":
            counts["final_generation"] += 1
        if purpose.endswith("_repair"):
            counts["repair_retries"] += 1
    return counts


def create_system():
    llm_client = OpenRouterClient()
    retrieval_pipeline = create_agent_retrieval_pipeline()
    sufficiency_checker = SufficiencyChecker(llm_client)
    evidence_adjudicator = EvidenceAdjudicator(llm_client)
    agent = AgentLoop(retrieval_pipeline, sufficiency_checker, evidence_adjudicator)
    answer_generator = AnswerGenerator(llm_client)

    return AlchemyXSystem(
        agent=agent,
        answer_generator=answer_generator,
    )


def ask(question):
    return create_system().ask(question)


def _notify(progress, message):
    if progress is None:
        return
    progress(message)
