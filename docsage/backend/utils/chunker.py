# """
# utils/chunker.py — Hierarchical semantic chunker.

# Addresses the 512-token BERT limitation (papers #14, #17, #23 in the review)
# with a two-level strategy:
#   - Level 1: Sentence-aware sliding window chunks (≤ chunk_size tokens)
#   - Level 2: For long documents, build a document-level summary chunk using SIF
#     (Smooth Inverse Frequency) weighted averaging — capturing global context

# Each chunk carries: content, chunk_id, doc_id, page, token_count, position,
# hierarchical_parent_id (for long-doc summary chunks).
# """
# from __future__ import annotations

# import re
# import hashlib
# from dataclasses import dataclass, field
# from typing import Optional

# import numpy as np
# from transformers import AutoTokenizer
# from core.config import settings
# from core.logging import get_logger
# from utils.document_parser import ParsedDocument, DocumentSection

# logger = get_logger(__name__)

# # Word-frequency table for SIF weighting (approximate — load real one in production)
# _SIF_A = 1e-3  # SIF smoothing constant


# @dataclass
# class Chunk:
#     """A text chunk ready for embedding and indexing."""
#     chunk_id: str
#     doc_id: str
#     content: str
#     token_count: int
#     page_number: Optional[int] = None
#     section_title: Optional[str] = None
#     section_type: str = "text"
#     chunk_index: int = 0          # position within document
#     total_chunks: int = 0
#     is_summary: bool = False      # True for hierarchical summary chunks
#     parent_chunk_ids: list[str] = field(default_factory=list)
#     metadata: dict = field(default_factory=dict)

#     def as_context_string(self) -> str:
#         """Format chunk with metadata prefix for LLM context."""
#         parts = []
#         if self.section_title:
#             parts.append(f"[Section: {self.section_title}]")
#         if self.page_number:
#             parts.append(f"[Page {self.page_number}]")
#         if self.is_summary:
#             parts.append("[Document Summary]")
#         parts.append(self.content)
#         return " ".join(parts)


# class HierarchicalChunker:
#     """
#     Two-level chunking strategy:

#     For any document:
#     1. Sentence-aware sliding window → fine-grained chunks
#     2. If total tokens > long_doc_threshold → create summary chunks per section
#        using SIF-weighted term importance (captures cross-512 context)

#     This directly addresses the LNLF-BERT and "Handling Long Sequences" papers
#     without requiring modified attention — works with any BERT variant.
#     """

#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             settings.reader_model_name,
#             cache_dir=str(settings.model_cache_dir),
#         )
#         self.chunk_size = settings.chunk_size_tokens
#         self.overlap = settings.chunk_overlap_tokens
#         self.long_threshold = settings.long_doc_threshold_tokens

#     def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
#         """Main entry point: chunk a full ParsedDocument."""
#         all_chunks: list[Chunk] = []
#         total_tokens = self._estimate_doc_tokens(doc)
#         is_long = total_tokens > self.long_threshold
#         chunk_index = 0

#         logger.info(
#             "chunking_document",
#             doc_id=doc.doc_id,
#             total_tokens=total_tokens,
#             is_long=is_long,
#         )

#         # Group sections by title for hierarchical grouping
#         section_groups = self._group_sections_by_heading(doc.sections)

#         for group_title, group_sections in section_groups.items():
#             group_chunks = []

#             for section in group_sections:
#                 section_chunks = self._chunk_section(
#                     section=section,
#                     doc_id=doc.doc_id,
#                     section_title=group_title,
#                     start_index=chunk_index,
#                 )
#                 group_chunks.extend(section_chunks)
#                 chunk_index += len(section_chunks)

#             all_chunks.extend(group_chunks)

#             # Build summary chunk for long documents
#             if is_long and len(group_chunks) >= 2:
#                 summary_chunk = self._build_summary_chunk(
#                     group_chunks=group_chunks,
#                     doc_id=doc.doc_id,
#                     section_title=group_title,
#                     chunk_index=chunk_index,
#                 )
#                 if summary_chunk:
#                     all_chunks.append(summary_chunk)
#                     chunk_index += 1

#         # Set total_chunks on all
#         total = len(all_chunks)
#         for c in all_chunks:
#             c.total_chunks = total

#         logger.info("chunking_complete", doc_id=doc.doc_id, num_chunks=total)
#         return all_chunks

#     def _chunk_section(
#         self,
#         section: DocumentSection,
#         doc_id: str,
#         section_title: Optional[str],
#         start_index: int,
#     ) -> list[Chunk]:
#         """Sliding window sentence-aware chunking of a section."""
#         if not section.content.strip():
#             return []

#         sentences = self._split_sentences(section.content)
#         if not sentences:
#             return []

#         chunks: list[Chunk] = []
#         current_sentences: list[str] = []
#         current_tokens: int = 0
#         local_idx = 0

#         for sent in sentences:
#             sent_tokens = self._count_tokens(sent)

#             # If adding this sentence exceeds limit, flush current buffer
#             if current_tokens + sent_tokens > self.chunk_size and current_sentences:
#                 chunk = self._make_chunk(
#                     sentences=current_sentences,
#                     doc_id=doc_id,
#                     page_number=section.page_number,
#                     section_title=section_title,
#                     section_type=section.section_type,
#                     chunk_index=start_index + local_idx,
#                 )
#                 chunks.append(chunk)
#                 local_idx += 1

#                 # Keep overlap sentences
#                 overlap_sentences = self._get_overlap_sentences(current_sentences)
#                 current_sentences = overlap_sentences + [sent]
#                 current_tokens = sum(self._count_tokens(s) for s in current_sentences)
#             else:
#                 current_sentences.append(sent)
#                 current_tokens += sent_tokens

#         # Flush remainder
#         if current_sentences:
#             chunk = self._make_chunk(
#                 sentences=current_sentences,
#                 doc_id=doc_id,
#                 page_number=section.page_number,
#                 section_title=section_title,
#                 section_type=section.section_type,
#                 chunk_index=start_index + local_idx,
#             )
#             chunks.append(chunk)

#         return chunks

#     def _build_summary_chunk(
#         self,
#         group_chunks: list[Chunk],
#         doc_id: str,
#         section_title: Optional[str],
#         chunk_index: int,
#     ) -> Optional[Chunk]:
#         """
#         Create a SIF-weighted summary of a group of chunks.

#         SIF (Smooth Inverse Frequency) assigns lower weight to very common words
#         and higher weight to rare/domain-specific terms — producing a better
#         representative sentence than simple averaging.

#         Here we apply it at the TEXT level: select the most informationally
#         dense sentences from across all chunks to form a summary.
#         """
#         all_sentences = []
#         for chunk in group_chunks:
#             all_sentences.extend(self._split_sentences(chunk.content))

#         if len(all_sentences) < 3:
#             return None

#         # Score sentences by term frequency × inverse document frequency proxy
#         word_freq = self._compute_word_freq(all_sentences)
#         scored = []
#         for sent in all_sentences:
#             words = re.findall(r'\w+', sent.lower())
#             if not words:
#                 continue
#             # SIF score: mean of a / (a + freq(word)) for each word
#             score = np.mean([
#                 _SIF_A / (_SIF_A + word_freq.get(w, _SIF_A))
#                 for w in words
#             ])
#             scored.append((score, sent))

#         # Select top sentences up to chunk_size tokens
#         scored.sort(reverse=True)
#         summary_sentences = []
#         summary_tokens = 0
#         for _, sent in scored:
#             t = self._count_tokens(sent)
#             if summary_tokens + t > self.chunk_size:
#                 break
#             summary_sentences.append(sent)
#             summary_tokens += t

#         if not summary_sentences:
#             return None

#         summary_text = " ".join(summary_sentences)
#         chunk_id = self._make_chunk_id(doc_id, f"summary_{section_title}_{chunk_index}")

#         return Chunk(
#             chunk_id=chunk_id,
#             doc_id=doc_id,
#             content=summary_text,
#             token_count=summary_tokens,
#             section_title=section_title,
#             section_type="text",
#             chunk_index=chunk_index,
#             is_summary=True,
#             parent_chunk_ids=[c.chunk_id for c in group_chunks],
#         )

#     def _compute_word_freq(self, sentences: list[str]) -> dict[str, float]:
#         """Compute normalized word frequency across all sentences."""
#         freq: dict[str, int] = {}
#         total = 0
#         for sent in sentences:
#             for word in re.findall(r'\w+', sent.lower()):
#                 freq[word] = freq.get(word, 0) + 1
#                 total += 1
#         if total == 0:
#             return {}
#         return {w: c / total for w, c in freq.items()}

#     def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
#         """Return last few sentences as overlap for the next chunk."""
#         overlap_tokens = 0
#         overlap_sents = []
#         for sent in reversed(sentences):
#             t = self._count_tokens(sent)
#             if overlap_tokens + t > self.overlap:
#                 break
#             overlap_sents.insert(0, sent)
#             overlap_tokens += t
#         return overlap_sents

#     def _split_sentences(self, text: str) -> list[str]:
#         """Simple but effective sentence splitter."""
#         # Split on sentence-ending punctuation followed by whitespace
#         sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
#         return [s.strip() for s in sentences if s.strip()]

#     def _count_tokens(self, text: str) -> int:
#         return len(self.tokenizer.encode(text, add_special_tokens=False))

#     def _estimate_doc_tokens(self, doc: ParsedDocument) -> int:
#         sample = doc.full_text[:5000]
#         ratio = self._count_tokens(sample) / max(len(sample), 1)
#         return int(ratio * len(doc.full_text))

#     def _group_sections_by_heading(
#         self, sections: list[DocumentSection]
#     ) -> dict[str, list[DocumentSection]]:
#         """Group consecutive sections under their most recent heading."""
#         groups: dict[str, list[DocumentSection]] = {}
#         current_heading = "Introduction"

#         for section in sections:
#             if section.section_type == "heading":
#                 current_heading = section.content[:80]
#                 if current_heading not in groups:
#                     groups[current_heading] = []
#             else:
#                 if current_heading not in groups:
#                     groups[current_heading] = []
#                 groups[current_heading].append(section)

#         return groups

#     def _make_chunk(
#         self,
#         sentences: list[str],
#         doc_id: str,
#         page_number: Optional[int],
#         section_title: Optional[str],
#         section_type: str,
#         chunk_index: int,
#     ) -> Chunk:
#         content = " ".join(sentences)
#         token_count = self._count_tokens(content)
#         chunk_id = self._make_chunk_id(doc_id, f"{chunk_index}_{content[:40]}")

#         return Chunk(
#             chunk_id=chunk_id,
#             doc_id=doc_id,
#             content=content,
#             token_count=token_count,
#             page_number=page_number,
#             section_title=section_title,
#             section_type=section_type,
#             chunk_index=chunk_index,
#         )

#     @staticmethod
#     def _make_chunk_id(doc_id: str, content_key: str) -> str:
#         h = hashlib.sha256(f"{doc_id}::{content_key}".encode()).hexdigest()[:12]
#         return f"chunk_{doc_id}_{h}"
"""
utils/chunker.py — Paragraph-level chunker for precise retrieval.

Key insight from the failing queries:
  "What are the two main modules?" → answer is in ONE paragraph
  "What threshold value was set?" → answer is in ONE sentence

Large chunks (400 tokens) bury these answers in noise. Small paragraph-level
chunks (100-200 tokens) let BM25 and dense retrieval find the exact paragraph.

Strategy:
  1. Split document into paragraphs (double newline boundaries)
  2. If a paragraph is short enough → one chunk
  3. If too long → sentence-level sliding window
  4. Always preserve page_number and section_title for citation
"""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoTokenizer
from core.config import settings
from core.logging import get_logger
from utils.document_parser import ParsedDocument, DocumentSection

logger = get_logger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    token_count: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    section_type: str = "text"
    chunk_index: int = 0
    total_chunks: int = 0
    is_summary: bool = False
    parent_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def as_context_string(self) -> str:
        """Used for dense embedding — include section for better semantic match."""
        parts = []
        if self.section_title:
            parts.append(self.section_title)
        parts.append(self.content)
        return " ".join(parts)


class HierarchicalChunker:
    """
    Paragraph-aware chunker.

    Produces small, focused chunks so retrieval can pinpoint the exact
    paragraph containing an answer — not bury it in a 400-token block.
    """

    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.reader_model_name,
                cache_dir=str(settings.model_cache_dir),
            )
        except Exception:
            # Fallback: approximate 1 token ≈ 0.75 words
            self.tokenizer = None

        self.chunk_size = settings.chunk_size_tokens
        self.overlap = settings.chunk_overlap_tokens

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        chunk_index = 0
        current_section = "Introduction"

        for section in doc.sections:
            if section.section_type == "heading":
                current_section = section.content.strip()[:80]
                continue

            if not section.content.strip():
                continue

            # Split section into paragraphs first
            paragraphs = self._split_paragraphs(section.content)

            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < 15:
                    continue

                para_tokens = self._count_tokens(para)

                if para_tokens <= self.chunk_size:
                    # Paragraph fits → one chunk
                    chunk = self._make_chunk(
                        content=para,
                        doc_id=doc.doc_id,
                        page_number=section.page_number,
                        section_title=current_section,
                        section_type=section.section_type,
                        index=chunk_index,
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1
                else:
                    # Paragraph too long → sentence sliding window
                    sub_chunks = self._sliding_window(
                        text=para,
                        doc_id=doc.doc_id,
                        page_number=section.page_number,
                        section_title=current_section,
                        section_type=section.section_type,
                        start_index=chunk_index,
                    )
                    all_chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)

        total = len(all_chunks)
        for c in all_chunks:
            c.total_chunks = total

        logger.info("chunking_complete", doc_id=doc.doc_id, num_chunks=total)
        return all_chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split on blank lines or clear paragraph boundaries."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Split on 2+ newlines
        parts = re.split(r"\n{2,}", text)
        # Also split on lines that look like new sentences after a period
        result = []
        for part in parts:
            # Further split very long single-line paragraphs on ". " boundaries
            sub = re.split(r"(?<=[.!?])\s{2,}", part)
            result.extend(sub)
        return result

    def _sliding_window(
        self,
        text: str,
        doc_id: str,
        page_number: Optional[int],
        section_title: Optional[str],
        section_type: str,
        start_index: int,
    ) -> list[Chunk]:
        sentences = self._split_sentences(text)
        chunks = []
        current: list[str] = []
        current_tokens = 0
        local_idx = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if sent_tokens > self.chunk_size:
                # Single sentence exceeds limit — hard truncate
                sent = self._truncate(sent, self.chunk_size)
                sent_tokens = self.chunk_size

            if current_tokens + sent_tokens > self.chunk_size and current:
                chunks.append(self._make_chunk(
                    content=" ".join(current),
                    doc_id=doc_id,
                    page_number=page_number,
                    section_title=section_title,
                    section_type=section_type,
                    index=start_index + local_idx,
                ))
                local_idx += 1
                # Keep overlap
                overlap_sents = self._overlap_sentences(current)
                current = overlap_sents + [sent]
                current_tokens = sum(self._count_tokens(s) for s in current)
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(self._make_chunk(
                content=" ".join(current),
                doc_id=doc_id,
                page_number=page_number,
                section_title=section_title,
                section_type=section_type,
                index=start_index + local_idx,
            ))

        return chunks

    def _overlap_sentences(self, sentences: list[str]) -> list[str]:
        kept = []
        tokens = 0
        for s in reversed(sentences):
            t = self._count_tokens(s)
            if tokens + t > self.overlap:
                break
            kept.insert(0, s)
            tokens += t
        return kept

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(])", text)
        return [p.strip() for p in parts if p.strip()]

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Approximate: 1 token ≈ 0.75 words
        return max(1, int(len(text.split()) * 1.3))

    def _truncate(self, text: str, max_tokens: int) -> str:
        if not self.tokenizer:
            words = text.split()
            return " ".join(words[: int(max_tokens / 1.3)])
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _make_chunk(
        self,
        content: str,
        doc_id: str,
        page_number: Optional[int],
        section_title: Optional[str],
        section_type: str,
        index: int,
    ) -> Chunk:
        token_count = self._count_tokens(content)
        key = f"{doc_id}:{index}:{content[:40]}"
        chunk_id = "chunk_" + hashlib.sha256(key.encode()).hexdigest()[:12]
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=content,
            token_count=token_count,
            page_number=page_number,
            section_title=section_title,
            section_type=section_type,
            chunk_index=index,
        )
