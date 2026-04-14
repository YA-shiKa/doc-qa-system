# DocSage — Smart Document QA System

A production-grade Question Answering system over documents combining:
- **Hybrid Retrieval**: Dense (FAISS + BERT embeddings) + Sparse (BM25) + Knowledge Graph
- **Long-Context Handling**: Hierarchical chunking beyond 512 tokens
- **Adversarial Robustness**: ELECTRA-based ensemble with confidence calibration
- **Multi-turn Conversations**: Full dialogue history with context tracking
- **Domain Adaptation**: Plug-and-play fine-tuning scaffold
- **REST API + React UI**: Fully deployable end-to-end

## Architecture

```
docsage/
├── backend/
│   ├── api/              # FastAPI routers (documents, qa, sessions)
│   ├── core/             # Pipeline engine, config, logging
│   ├── models/           # BERT reader, ELECTRA robustness, embedder, reranker
│   ├── utils/            # Chunker, retriever, KG builder, PDF parser
│   └── tests/            # Unit + integration tests
├── frontend/
│   └── src/
│       ├── components/   # ChatInterface, DocumentUploader, AnswerCard, etc.
│       ├── hooks/        # useQA, useDocuments, useSession
│       ├── pages/        # Home, Session, Library
│       └── services/     # API client
├── scripts/              # Indexing, fine-tuning, eval scripts
├── configs/              # Model configs, retrieval configs
└── docker/               # Dockerfiles + compose
```

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
python -m uvicorn api.main:app --reload

# Frontend
cd frontend
npm install && npm run dev

# Docker (full stack)
docker compose -f docker/docker-compose.yml up
```

## Key Innovations vs Literature

| Paper Limitation | DocSage Solution |
|---|---|
| Single retrieval method | Hybrid BM25 + FAISS + KG fusion |
| 512-token limit | Hierarchical chunking + SIF aggregation |
| Single model vulnerability | ELECTRA ensemble + adversarial filter |
| No confidence signals | Calibrated probability + evidence citation |
| Static context | Multi-turn conversational memory |
| Domain-specific only | Universal + domain fine-tuning scaffold |