# ADR-001: Local LLM Model Selection

## Status

Accepted

## Date

2026-10-07

## Context

The Enterprise Knowledge Intelligence Platform uses a local large language
model (LLM) for grounded answer generation.

Two locally available models were evaluated:

- Llama 3.2 3B
- Gemma 2 9B

The objective was to select a model for the current local deployment based
on measured application behavior rather than model size alone.

The evaluation used the same:

- source document
- document ingestion pipeline
- chunking configuration
- embedding model
- FAISS vector store
- retrieval configuration
- evidence-selection thresholds
- prompts
- temperature
- evaluation questions

Only the generation model was changed.

## Evaluation Dataset

Six representative RAG scenarios were evaluated:

1. Direct factual question
2. Semantic/indirect question
3. Contextual question
4. Programming-related question
5. Document-location question
6. Unanswerable question requiring abstention

## Results

| Metric | Llama 3.2 3B | Gemma 2 9B |
|---|---:|---:|
| Overall pass rate | 6/6 (100%) | 6/6 (100%) |
| Answerable accuracy | 5/5 (100%) | 5/5 (100%) |
| Abstention accuracy | 1/1 (100%) | 1/1 (100%) |
| Average retrieval latency | 40.32 ms | 23.31 ms |
| Average generation latency | 7.14 s | 20.73 s |
| Average end-to-end latency | 7.18 s | 20.75 s |

The evaluation therefore showed equivalent functional performance on the
current test set.

Gemma 2 9B had lower measured retrieval latency in this run, but retrieval
uses the same embedding and FAISS components for both models. Therefore this
difference is treated as runtime variation rather than a model-selection
advantage.

Llama 3.2 3B generated answers substantially faster on the tested workload.

## Decision

Use **Llama 3.2 3B** as the default local generation model.

The model is configured externally through the `RAG_MODEL` environment
variable rather than being hard-coded into the application.

Current configuration:

```text
RAG_MODEL=llama3.2:3b