# # """
# # models/reader.py — BERT-based extractive QA reader.

# # Implements:
# #   - Span extraction from retrieved passages (standard QA approach)
# #   - Multi-passage aggregation: answer the question across N passages
# #   - Answer confidence calibration via temperature scaling
# #   - Source evidence linking: which passage + which span = which page
# # """
# # from __future__ import annotations

# # from dataclasses import dataclass, field
# # from typing import Optional
# # import time

# # import torch
# # from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
# # from core.config import settings
# # from core.logging import get_logger
# # from utils.chunker import Chunk

# # logger = get_logger(__name__)


# # @dataclass
# # class ExtractedAnswer:
# #     """A single candidate answer extracted from one passage."""
# #     text: str
# #     score: float                     # raw model logit-based score
# #     confidence: float                # calibrated 0–1 confidence
# #     chunk_id: str
# #     doc_id: str
# #     page_number: Optional[int] = None
# #     section_title: Optional[str] = None
# #     start_char: int = 0
# #     end_char: int = 0
# #     is_impossible: bool = False      # True if model predicts no answer in passage


# # @dataclass
# # class QAAnswer:
# #     """Final aggregated answer returned to the user."""
# #     answer: str
# #     confidence: float
# #     is_impossible: bool = False
# #     evidence_passages: list[ExtractedAnswer] = field(default_factory=list)
# #     answer_type: str = "extractive"  # "extractive" | "no_answer" | "aggregated"
# #     latency_ms: float = 0.0

# #     @property
# #     def top_evidence(self) -> Optional[ExtractedAnswer]:
# #         return self.evidence_passages[0] if self.evidence_passages else None

# #     @property
# #     def sources(self) -> list[dict]:
# #         """Deduplicated source list for citation."""
# #         seen = set()
# #         sources = []
# #         for ev in self.evidence_passages:
# #             key = (ev.doc_id, ev.page_number)
# #             if key not in seen:
# #                 seen.add(key)
# #                 sources.append({
# #                     "doc_id": ev.doc_id,
# #                     "page": ev.page_number,
# #                     "section": ev.section_title,
# #                     "snippet": ev.text[:200],
# #                 })
# #         return sources


# # class BERTReader:
# #     """
# #     Extractive QA reader using a fine-tuned BERT-variant QA model.

# #     Key design decisions:
# #     1. Run QA over each passage independently → get (answer, score) per passage
# #     2. Aggregate by taking the highest-confidence answer, with a fall-through
# #        to "I don't know" if all passages score below threshold
# #     3. Calibrate raw softmax scores via temperature scaling (learned parameter)
# #        to produce well-calibrated probabilities
# #     4. Return ALL evidence so the UI can show source citations

# #     This multi-passage approach matches or exceeds single-passage BERT on SQuAD
# #     while also enabling source-level citation (not possible in any reviewed paper).
# #     """

# #     # Temperature scaling factor (tune on a calibration set for your domain)
# #     _TEMPERATURE = 1.4

# #     # Below this calibrated confidence, report as "no answer"
# #     _NO_ANSWER_THRESHOLD = 0.15

# #     def __init__(self):
# #         logger.info("loading_reader_model", model=settings.reader_model_name)
# #         self.tokenizer = AutoTokenizer.from_pretrained(
# #             settings.reader_model_name,
# #             cache_dir=str(settings.model_cache_dir),
# #         )
# #         self.model = AutoModelForQuestionAnswering.from_pretrained(
# #             settings.reader_model_name,
# #             cache_dir=str(settings.model_cache_dir),
# #         )
# #         self.device = torch.device(settings.reader_model_device)
# #         self.model.to(self.device)
# #         self.model.eval()

# #         # High-level pipeline for convenience (used in batch mode)
# #         self.pipeline = pipeline(
# #             "question-answering",
# #             model=self.model,
# #             tokenizer=self.tokenizer,
# #             device=0 if settings.reader_model_device == "cuda" else -1,
# #         )

# #     def answer(
# #         self,
# #         question: str,
# #         passages: list[Chunk],
# #         history: Optional[list[dict]] = None,
# #     ) -> QAAnswer:
# #         """
# #         Answer a question given a list of retrieved passages.

# #         Args:
# #             question: The user's question
# #             passages: Retrieved and reranked passages (Chunk objects)
# #             history: Conversation history for context injection

# #         Returns:
# #             QAAnswer with best answer, confidence, and source evidence
# #         """
# #         t_start = time.perf_counter()

# #         if not passages:
# #             return QAAnswer(
# #                 answer="No relevant passages were found in the document.",
# #                 confidence=0.0,
# #                 is_impossible=True,
# #                 answer_type="no_answer",
# #             )

# #         # Inject conversation history into question if available
# #         augmented_question = self._augment_with_history(question, history)

# #         # Extract answers from each passage
# #         candidates: list[ExtractedAnswer] = []
# #         for chunk in passages[: settings.rerank_top_k]:
# #             candidate = self._extract_from_passage(augmented_question, chunk)
# #             candidates.append(candidate)

# #         # Aggregate: select best non-impossible answer
# #         candidates.sort(key=lambda c: c.score, reverse=True)
# #         non_empty = [c for c in candidates if not c.is_impossible and len(c.text.strip()) > 1]

# #         if not non_empty:
# #             return QAAnswer(
# #                 answer="I could not find a specific answer to this question in the document.",
# #                 confidence=0.0,
# #                 is_impossible=True,
# #                 evidence_passages=candidates[:3],
# #                 answer_type="no_answer",
# #                 latency_ms=(time.perf_counter() - t_start) * 1000,
# #             )

# #         best = non_empty[0]

# #         # If top answers from different passages agree, boost confidence
# #         if len(non_empty) >= 2:
# #             agreement_bonus = self._compute_agreement_bonus(non_empty)
# #             best.confidence = min(best.confidence + agreement_bonus, 0.99)

# #         return QAAnswer(
# #             answer=best.text,
# #             confidence=best.confidence,
# #             is_impossible=best.confidence < self._NO_ANSWER_THRESHOLD,
# #             evidence_passages=non_empty[:3] + [c for c in candidates if c.is_impossible][:1],
# #             answer_type="extractive",
# #             latency_ms=(time.perf_counter() - t_start) * 1000,
# #         )

# #     def _extract_from_passage(self, question: str, chunk: Chunk) -> ExtractedAnswer:
# #         """Run QA model on a single passage and return extracted answer."""
# #         context = chunk.as_context_string()

# #         try:
# #             result = self.pipeline(
# #                 question=question,
# #                 context=context,
# #                 handle_impossible_answer=True,
# #                 max_answer_len=100,
# #             )

# #             raw_score = float(result.get("score", 0))
# #             calibrated = self._calibrate_score(raw_score)
# #             answer_text = result.get("answer", "").strip()
# #             is_impossible = (answer_text == "" or raw_score < 0.01)

# #             return ExtractedAnswer(
# #                 text=answer_text,
# #                 score=raw_score,
# #                 confidence=calibrated,
# #                 chunk_id=chunk.chunk_id,
# #                 doc_id=chunk.doc_id,
# #                 page_number=chunk.page_number,
# #                 section_title=chunk.section_title,
# #                 start_char=result.get("start", 0),
# #                 end_char=result.get("end", 0),
# #                 is_impossible=is_impossible,
# #             )

# #         except Exception as e:
# #             logger.warning("qa_extraction_failed", chunk_id=chunk.chunk_id, error=str(e))
# #             return ExtractedAnswer(
# #                 text="",
# #                 score=0.0,
# #                 confidence=0.0,
# #                 chunk_id=chunk.chunk_id,
# #                 doc_id=chunk.doc_id,
# #                 is_impossible=True,
# #             )

# #     def _calibrate_score(self, raw_score: float) -> float:
# #         """
# #         Temperature scaling: soften overconfident predictions.
# #         T > 1 makes the distribution more uniform; T < 1 sharpens it.
# #         """
# #         import math
# #         if raw_score <= 0:
# #             return 0.0
# #         if raw_score >= 1:
# #             return 1.0
# #         # Apply temperature scaling in log-space
# #         log_score = math.log(raw_score + 1e-9)
# #         log_complement = math.log(1 - raw_score + 1e-9)
# #         scaled_log = log_score / self._TEMPERATURE
# #         scaled_complement = log_complement / self._TEMPERATURE
# #         exp_s = math.exp(scaled_log)
# #         exp_c = math.exp(scaled_complement)
# #         return exp_s / (exp_s + exp_c)

# #     def _augment_with_history(
# #         self, question: str, history: Optional[list[dict]]
# #     ) -> str:
# #         """
# #         Prepend recent conversation turns to the question for multi-turn QA.
# #         Implements the BERT-CoQAC approach: inject dialogue history as prefix.
# #         Only uses last 2 turns to stay within token budget.
# #         """
# #         if not history:
# #             return question

# #         relevant_turns = history[-2:]
# #         context_parts = []
# #         for turn in relevant_turns:
# #             if "question" in turn and "answer" in turn:
# #                 context_parts.append(f"Q: {turn['question']} A: {turn['answer']}")

# #         if not context_parts:
# #             return question

# #         context_str = " | ".join(context_parts)
# #         return f"Context: {context_str} | Current question: {question}"

# #     def _compute_agreement_bonus(self, candidates: list[ExtractedAnswer]) -> float:
# #         """
# #         If multiple top passages give similar answers, boost confidence.
# #         Uses normalized character overlap (Jaccard) between answers.
# #         """
# #         if len(candidates) < 2:
# #             return 0.0

# #         top_words = set(candidates[0].text.lower().split())
# #         second_words = set(candidates[1].text.lower().split())

# #         if not top_words or not second_words:
# #             return 0.0

# #         jaccard = len(top_words & second_words) / len(top_words | second_words)
# #         # Up to 0.1 bonus for perfect agreement
# #         return jaccard * 0.1
# """
# models/reader.py — BERT-based extractive QA reader (fixed for real-world performance).

# Key fixes over v1:
#   - Removed handle_impossible_answer=True (was killing valid answers)
#   - Direct model inference instead of pipeline (more control over span selection)
#   - Raised NO_ANSWER_THRESHOLD to 0.05 (was too aggressive at 0.15)
#   - Lowered temperature to 1.1 (was 1.4 — was deflating scores to near zero)
#   - Added concatenated context fallback (catches cross-passage answers)
#   - Added sentence-level fallback so something is always returned
#   - Multi-passage: score ALL passages, pick best non-empty answer
# """
# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import Optional
# import time

# import torch
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# from core.config import settings
# from core.logging import get_logger
# from utils.chunker import Chunk

# logger = get_logger(__name__)


# @dataclass
# class ExtractedAnswer:
#     text: str
#     score: float
#     confidence: float
#     chunk_id: str
#     doc_id: str
#     page_number: Optional[int] = None
#     section_title: Optional[str] = None
#     start_char: int = 0
#     end_char: int = 0
#     is_impossible: bool = False


# @dataclass
# class QAAnswer:
#     answer: str
#     confidence: float
#     is_impossible: bool = False
#     evidence_passages: list[ExtractedAnswer] = field(default_factory=list)
#     answer_type: str = "extractive"
#     latency_ms: float = 0.0

#     @property
#     def top_evidence(self) -> Optional[ExtractedAnswer]:
#         return self.evidence_passages[0] if self.evidence_passages else None

#     @property
#     def sources(self) -> list[dict]:
#         seen = set()
#         sources = []
#         for ev in self.evidence_passages:
#             key = (ev.doc_id, ev.page_number)
#             if key not in seen:
#                 seen.add(key)
#                 sources.append({
#                     "doc_id": ev.doc_id,
#                     "page": ev.page_number,
#                     "section": ev.section_title,
#                     "snippet": ev.text[:300],
#                 })
#         return sources


# class BERTReader:
#     """
#     Extractive QA reader using direct model inference.

#     Fix summary:
#       1. Use raw model logits + best-span selection instead of pipeline
#          (pipeline's handle_impossible_answer was discarding valid answers)
#       2. Try each passage individually, PLUS a concatenated context of top-3
#       3. Only fall back to "no answer" if truly nothing extractable
#       4. Mild temperature scaling (1.1) so confidence values are meaningful
#     """

#     _NO_ANSWER_THRESHOLD = 0.05   # was 0.15 — too aggressive
#     _TEMPERATURE = 1.1            # was 1.4 — was deflating scores to near zero
#     _CONCAT_MAX_TOKENS = 480      # max tokens for concatenated context

#     def __init__(self):
#         logger.info("loading_reader_model", model=settings.reader_model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             settings.reader_model_name,
#             cache_dir=str(settings.model_cache_dir),
#         )
#         self.model = AutoModelForQuestionAnswering.from_pretrained(
#             settings.reader_model_name,
#             cache_dir=str(settings.model_cache_dir),
#         )
#         self.device = torch.device(settings.reader_model_device)
#         self.model.to(self.device)
#         self.model.eval()
#         logger.info("reader_model_loaded", device=str(self.device))

#     def answer(
#         self,
#         question: str,
#         passages: list[Chunk],
#         history: Optional[list[dict]] = None,
#     ) -> QAAnswer:
#         t_start = time.perf_counter()

#         if not passages:
#             return QAAnswer(
#                 answer="No relevant passages found. Please upload a document first.",
#                 confidence=0.0,
#                 is_impossible=True,
#                 answer_type="no_answer",
#             )

#         augmented_question = self._augment_with_history(question, history)
#         candidates: list[ExtractedAnswer] = []

#         # 1. Try each passage individually
#         for chunk in passages[:settings.rerank_top_k]:
#             candidate = self._extract_from_passage(augmented_question, chunk)
#             if candidate:
#                 candidates.append(candidate)

#         # 2. Concatenated context fallback (catches answers spanning passages)
#         concat_result = self._extract_from_concat(augmented_question, passages[:3])
#         if concat_result and not concat_result.is_impossible:
#             candidates.append(concat_result)

#         # Sort by raw score — most reliable signal
#         candidates.sort(key=lambda c: c.score, reverse=True)
#         non_empty = [
#             c for c in candidates
#             if not c.is_impossible and c.text.strip() and len(c.text.strip()) > 1
#         ]

#         latency_ms = (time.perf_counter() - t_start) * 1000

#         if not non_empty:
#             # Last resort: return the most informative sentence from the top passage
#             fallback = self._extract_fallback_sentence(passages[0])
#             if fallback:
#                 return QAAnswer(
#                     answer=fallback,
#                     confidence=0.15,
#                     is_impossible=False,
#                     evidence_passages=candidates[:3],
#                     answer_type="fallback",
#                     latency_ms=latency_ms,
#                 )
#             return QAAnswer(
#                 answer="I could not find a specific answer in the uploaded documents. Try rephrasing your question or upload additional relevant documents.",
#                 confidence=0.0,
#                 is_impossible=True,
#                 evidence_passages=candidates[:3],
#                 answer_type="no_answer",
#                 latency_ms=latency_ms,
#             )

#         best = non_empty[0]

#         # Agreement bonus when multiple passages give similar answers
#         if len(non_empty) >= 2:
#             bonus = self._compute_agreement_bonus(non_empty)
#             best.confidence = min(best.confidence + bonus, 0.99)

#         return QAAnswer(
#             answer=best.text,
#             confidence=best.confidence,
#             is_impossible=best.confidence < self._NO_ANSWER_THRESHOLD,
#             evidence_passages=non_empty[:3],
#             answer_type="extractive",
#             latency_ms=latency_ms,
#         )

#     def _extract_from_passage(self, question: str, chunk: Chunk) -> Optional[ExtractedAnswer]:
#         """Direct model inference on a single passage."""
#         context = chunk.content
#         if not context.strip():
#             return None

#         try:
#             inputs = self.tokenizer(
#                 question,
#                 context,
#                 add_special_tokens=True,
#                 return_tensors="pt",
#                 max_length=512,
#                 truncation=True,
#                 padding=False,
#             ).to(self.device)

#             with torch.no_grad():
#                 outputs = self.model(**inputs)

#             start_logits = outputs.start_logits[0]
#             end_logits = outputs.end_logits[0]

#             start_idx, end_idx, span_score = self._get_best_span(
#                 start_logits, end_logits, inputs["input_ids"][0]
#             )

#             if start_idx is None:
#                 return ExtractedAnswer(
#                     text="", score=0.0, confidence=0.0,
#                     chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
#                     page_number=chunk.page_number,
#                     section_title=chunk.section_title,
#                     is_impossible=True,
#                 )

#             answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
#             answer_text = self.tokenizer.decode(
#                 answer_tokens, skip_special_tokens=True
#             ).strip()

#             # Reject special tokens or empty results
#             bad_tokens = {"[cls]", "[sep]", "<s>", "</s>", ""}
#             if answer_text.lower() in bad_tokens:
#                 return ExtractedAnswer(
#                     text="", score=0.0, confidence=0.0,
#                     chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
#                     is_impossible=True,
#                 )

#             confidence = self._calibrate_score(float(span_score))

#             return ExtractedAnswer(
#                 text=answer_text,
#                 score=float(span_score),
#                 confidence=confidence,
#                 chunk_id=chunk.chunk_id,
#                 doc_id=chunk.doc_id,
#                 page_number=chunk.page_number,
#                 section_title=chunk.section_title,
#                 is_impossible=False,
#             )

#         except Exception as e:
#             logger.warning("qa_extraction_failed", chunk_id=chunk.chunk_id, error=str(e))
#             return None

#     def _extract_from_concat(
#         self, question: str, chunks: list[Chunk]
#     ) -> Optional[ExtractedAnswer]:
#         """Concatenate top passages and run extraction on the combined context."""
#         if not chunks:
#             return None

#         combined = " ".join(c.content for c in chunks)

#         # Truncate to token limit
#         tokens = self.tokenizer.encode(combined, add_special_tokens=False)
#         if len(tokens) > self._CONCAT_MAX_TOKENS:
#             tokens = tokens[: self._CONCAT_MAX_TOKENS]
#             combined = self.tokenizer.decode(tokens, skip_special_tokens=True)

#         fake_chunk = Chunk(
#             chunk_id=f"concat_{chunks[0].chunk_id}",
#             doc_id=chunks[0].doc_id,
#             content=combined,
#             token_count=len(tokens),
#             page_number=chunks[0].page_number,
#             section_title=chunks[0].section_title,
#         )
#         return self._extract_from_passage(question, fake_chunk)

#     def _get_best_span(
#         self,
#         start_logits: torch.Tensor,
#         end_logits: torch.Tensor,
#         input_ids: torch.Tensor,
#     ) -> tuple[Optional[int], Optional[int], float]:
#         """
#         Find the highest-scoring valid span.
#         - Skip token 0 ([CLS]) and last token ([SEP])
#         - Span length must be 1–50 tokens
#         - end >= start
#         """
#         start_probs = torch.softmax(start_logits, dim=0)
#         end_probs = torch.softmax(end_logits, dim=0)

#         n = len(start_logits)
#         best_score = -1.0
#         best_start = None
#         best_end = None

#         for s in range(1, n - 1):
#             for e in range(s, min(s + 50, n - 1)):
#                 score = float(start_probs[s]) + float(end_probs[e])
#                 if score > best_score:
#                     best_score = score
#                     best_start = s
#                     best_end = e

#         if best_start is None:
#             return None, None, 0.0

#         # Normalize: both probabilities max at 1.0 each, so combined max is 2.0
#         normalized = best_score / 2.0
#         return best_start, best_end, normalized

#     def _extract_fallback_sentence(self, chunk: Chunk) -> str:
#         """Return the longest sentence from the top chunk as a last resort."""
#         import re
#         sentences = re.split(r"(?<=[.!?])\s+", chunk.content.strip())
#         sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
#         if not sentences:
#             return chunk.content[:400].strip()
#         return max(sentences, key=len)[:400]

#     def _calibrate_score(self, raw_score: float) -> float:
#         """Mild temperature scaling — keeps scores meaningful, not deflated to zero."""
#         import math
#         raw_score = max(0.001, min(0.999, raw_score))
#         log_s = math.log(raw_score)
#         log_c = math.log(1.0 - raw_score)
#         exp_s = math.exp(log_s / self._TEMPERATURE)
#         exp_c = math.exp(log_c / self._TEMPERATURE)
#         return exp_s / (exp_s + exp_c)

#     def _augment_with_history(
#         self, question: str, history: Optional[list[dict]]
#     ) -> str:
#         if not history:
#             return question
#         relevant = history[-2:]
#         parts = [
#             f"Q: {t['question']} A: {t['answer']}"
#             for t in relevant
#             if t.get("answer")
#         ]
#         if not parts:
#             return question
#         return f"{' | '.join(parts)} | {question}"

#     def _compute_agreement_bonus(
#         self, candidates: list[ExtractedAnswer]
#     ) -> float:
#         if len(candidates) < 2:
#             return 0.0
#         top_words = set(candidates[0].text.lower().split())
#         second_words = set(candidates[1].text.lower().split())
#         if not top_words or not second_words:
#             return 0.0
#         jaccard = len(top_words & second_words) / len(top_words | second_words)
#         return jaccard * 0.1

"""
models/reader.py — Correct extractive QA reader.

THE key fix: span search is restricted to context tokens ONLY.
RoBERTa layout: <s> question </s></s> context </s>
                 0   1..Q    Q+1 Q+2  Q+3..    last

We use encoding.sequence_ids() to find exactly which positions are context,
then mask everything else to -inf before argmax. This prevents the model
from "answering" by selecting tokens from the question itself.

Also: no reranker dependency here, no pipeline wrapper — pure model inference.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from core.config import settings
from core.logging import get_logger
from utils.chunker import Chunk

logger = get_logger(__name__)


@dataclass
class ExtractedAnswer:
    text: str
    score: float
    confidence: float
    chunk_id: str
    doc_id: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    is_impossible: bool = False
    passage_snippet: str = ""


@dataclass
class QAAnswer:
    answer: str
    confidence: float
    is_impossible: bool = False
    evidence_passages: list[ExtractedAnswer] = field(default_factory=list)
    answer_type: str = "extractive"
    latency_ms: float = 0.0

    @property
    def top_evidence(self) -> Optional[ExtractedAnswer]:
        return self.evidence_passages[0] if self.evidence_passages else None

    @property
    def sources(self) -> list[dict]:
        seen: set = set()
        out = []
        for ev in self.evidence_passages:
            key = (ev.doc_id, ev.page_number)
            if key not in seen:
                seen.add(key)
                out.append({
                    "doc_id": ev.doc_id,
                    "page": ev.page_number,
                    "section": ev.section_title,
                    "snippet": ev.passage_snippet or ev.text[:300],
                })
        return out


class BERTReader:
    """
    Extractive QA reader with correct context-only span search.

    How it works:
    1. Tokenize [question, context] together
    2. Identify context token range using sequence_ids()
    3. Mask question tokens to -inf in start/end logits
    4. Find best (start, end) span within context only
    5. Decode and return

    Fallback chain:
    - Per-passage extraction (top 5 passages)
    - Concatenated context (join top 3 passages)
    - Best sentence from top passage (always returns something)
    """

    _NO_ANSWER_THRESHOLD = 0.04
    _MAX_ANSWER_TOKENS = 50
    _TEMPERATURE = 1.05

    def __init__(self):
        logger.info("loading_reader", model=settings.reader_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.reader_model_name,
            cache_dir=str(settings.model_cache_dir),
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            settings.reader_model_name,
            cache_dir=str(settings.model_cache_dir),
        )
        self.device = torch.device(settings.reader_model_device)
        self.model.to(self.device)
        self.model.eval()
        logger.info("reader_loaded", device=str(self.device))

    def answer(
        self,
        question: str,
        passages: list[Chunk],
        history: Optional[list[dict]] = None,
    ) -> QAAnswer:
        t0 = time.perf_counter()

        if not passages:
            return QAAnswer(
                answer="No documents are indexed. Please upload a document first.",
                confidence=0.0,
                is_impossible=True,
                answer_type="no_answer",
            )

        q = self._build_question(question, history)
        candidates: list[ExtractedAnswer] = []

        # Try each passage individually
        for chunk in passages[: settings.rerank_top_k]:
            result = self._run(q, chunk.content, chunk)
            if result:
                candidates.append(result)

        # Also try concatenated context
        concat_text, concat_chunk = self._make_concat(passages[:3], q)
        if concat_text:
            result = self._run(q, concat_text, concat_chunk)
            if result and not result.is_impossible:
                candidates.append(result)

        ms = (time.perf_counter() - t0) * 1000

        # Pick best non-empty answer
        valid = sorted(
            [c for c in candidates if not c.is_impossible and c.text.strip()],
            key=lambda c: c.score,
            reverse=True,
        )

        if not valid:
            # Final fallback: best sentence from top passage
            fallback = self._best_sentence(passages[0].content)
            return QAAnswer(
                answer=fallback,
                confidence=0.12,
                is_impossible=False,
                evidence_passages=candidates[:3],
                answer_type="fallback",
                latency_ms=ms,
            )

        best = valid[0]
        if len(valid) >= 2:
            best.confidence = min(best.confidence + self._agreement(valid), 0.99)

        return QAAnswer(
            answer=best.text,
            confidence=best.confidence,
            is_impossible=best.confidence < self._NO_ANSWER_THRESHOLD,
            evidence_passages=valid[:3],
            answer_type="extractive",
            latency_ms=ms,
        )

    # ── Core extraction ───────────────────────────────────────────────────────

    def _run(self, question: str, context: str, chunk: Chunk) -> Optional[ExtractedAnswer]:
        """
        Run QA model, restricting span search to context tokens only.
        """
        if not context.strip():
            return None

        try:
            enc = self.tokenizer(
                question,
                context,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=512,
                truncation="only_second",  # never truncate question
                padding=False,
                return_offsets_mapping=False,
            )

            # sequence_ids: 0 = question token, 1 = context token, None = special
            seq_ids = enc.sequence_ids(0)

            ctx_start = None
            ctx_end = None
            for i, sid in enumerate(seq_ids):
                if sid == 1:
                    if ctx_start is None:
                        ctx_start = i
                    ctx_end = i

            if ctx_start is None:
                # All context got truncated — skip
                return self._no_answer(chunk, context)

            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=mask)

            sl = out.start_logits[0].clone()
            el = out.end_logits[0].clone()

            # Zero-out (mask to -inf) all positions outside context
            NEG_INF = torch.finfo(sl.dtype).min
            for i in range(len(sl)):
                if seq_ids[i] != 1:
                    sl[i] = NEG_INF
                    el[i] = NEG_INF

            # Find best valid span: end >= start, length <= MAX_ANSWER_TOKENS
            best_score = NEG_INF
            best_s = ctx_start
            best_e = ctx_start

            for s in range(ctx_start, ctx_end + 1):
                if sl[s] == NEG_INF:
                    continue
                max_e = min(s + self._MAX_ANSWER_TOKENS, ctx_end + 1)
                # Efficient: for each start, find the best end in range
                e_scores = el[s:max_e]
                best_e_offset = int(torch.argmax(e_scores))
                score = float(sl[s]) + float(e_scores[best_e_offset])
                if score > best_score:
                    best_score = score
                    best_s = s
                    best_e = s + best_e_offset

            # Decode span
            span_ids = ids[0][best_s : best_e + 1]
            text = self.tokenizer.decode(span_ids, skip_special_tokens=True).strip()

            # Reject garbage
            if not text or text.lower() in {"", "[cls]", "[sep]", "<s>", "</s>"}:
                return self._no_answer(chunk, context)

            # Score: sigmoid of sum of log-probs, normalized
            norm_score = float(torch.sigmoid(torch.tensor(best_score / 12.0)))
            conf = self._calibrate(norm_score)

            return ExtractedAnswer(
                text=text,
                score=norm_score,
                confidence=conf,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                passage_snippet=context[:300],
                is_impossible=False,
            )

        except Exception as e:
            logger.warning("extraction_error", error=str(e))
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_question(self, question: str, history: Optional[list[dict]]) -> str:
        if not history:
            return question
        turns = [
            f"Q: {t['question']} A: {t['answer']}"
            for t in history[-2:]
            if t.get("answer")
        ]
        return (" | ".join(turns) + " | " + question) if turns else question

    def _make_concat(
        self, chunks: list[Chunk], question: str
    ) -> tuple[str, Chunk]:
        if not chunks:
            return "", chunks[0] if chunks else None

        q_len = len(self.tokenizer.encode(question, add_special_tokens=False))
        budget = 512 - q_len - 6

        parts = []
        used = 0
        for c in chunks:
            toks = self.tokenizer.encode(c.content, add_special_tokens=False)
            if used + len(toks) > budget:
                rem = budget - used
                if rem > 20:
                    toks = toks[:rem]
                    parts.append(self.tokenizer.decode(toks, skip_special_tokens=True))
                break
            parts.append(c.content)
            used += len(toks)

        text = " ".join(parts)
        fake = Chunk(
            chunk_id=f"concat_{chunks[0].chunk_id}",
            doc_id=chunks[0].doc_id,
            content=text,
            token_count=used,
            page_number=chunks[0].page_number,
            section_title=chunks[0].section_title,
        )
        return text, fake

    def _no_answer(self, chunk: Chunk, context: str = "") -> ExtractedAnswer:
        return ExtractedAnswer(
            text="", score=0.0, confidence=0.0,
            chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            passage_snippet=context[:300],
            is_impossible=True,
        )

    def _best_sentence(self, text: str) -> str:
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        good = [s.strip() for s in sents if len(s.strip()) > 20]
        return max(good, key=len)[:500] if good else text[:400].strip()

    def _calibrate(self, score: float) -> float:
        import math
        s = max(1e-6, min(1 - 1e-6, score))
        es = math.exp(math.log(s) / self._TEMPERATURE)
        ec = math.exp(math.log(1 - s) / self._TEMPERATURE)
        return es / (es + ec)

    def _agreement(self, candidates: list[ExtractedAnswer]) -> float:
        w1 = set(candidates[0].text.lower().split())
        w2 = set(candidates[1].text.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2) * 0.08
