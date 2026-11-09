# kernel-rag-dotnet

A RAG assistant in C# on ASP.NET Core (.NET 8), built with Microsoft Semantic Kernel and local Ollama. No cloud services.
One `IVectorStore` interface has real implementations for Qdrant, Weaviate and Elasticsearch, and an evaluation compares them on identical data.

## Results

**Not measured yet.** The evaluation has not run, so this repository makes no claim about which store is better.
Numbers will appear in `evals/results/latest.json` after the `eval` GitHub Actions workflow runs. The threshold in
`evals/thresholds.json` (recall@5 of 0.5) is a placeholder until a first measurement exists.

## Architecture

```
Browser UI / HTTP --> ASP.NET Core minimal API --> RagService (Semantic Kernel)
                                                     |-- Ollama: nomic-embed-text, qwen2.5:1.5b
                                                     '-- IVectorStore: Qdrant | Weaviate | Elasticsearch | exact in-memory
```

- `src/KernelRag.Core`: RAG pipeline, chunker, metrics, groundedness scoring, store implementations.
- `src/KernelRag.Api`: `POST /api/ingest`, `POST /api/ask`, `GET /api/health`, and a small web page.
- `src/KernelRag.Eval`: data preparation and the store comparison.
- `tests/KernelRag.Tests`: xUnit unit and API tests, plus contract tests that run against real containers in CI.

The stores are called over their REST APIs through one shared `HttpClient` path, rather than through three different client libraries, so client differences do not affect the latency comparison.

## How the comparison is designed

- Every store receives the same vectors, embedded once.
- All use cosine HNSW with m=16, ef_construction=100, ef_search=100, without quantization or keyword indexing.
- Qdrant's thresholds are lowered, because its defaults would answer this small collection by exact scan.
- Indexing is timed over 3 trials, up to the point the index is searchable. The median is reported.
- 50 warm-up queries are excluded. Stores are queried in rotating order per question, to spread runner drift evenly.
- An exact in-memory search provides the recall ceiling for these embeddings. Its latency is not comparable, because it makes no network call.
- Groundedness (`workflow_dispatch` only): `qwen2.5:1.5b` answers from the top 5 passages and `qwen2.5:3b` judges support. A control judges answers against unrelated context, to show how far the judge can be trusted. A lexical score and gold-answer containment are reported alongside it.

## Data

`data/` holds SQuAD 1.1 dev: 2,067 Wikipedia passages and 500 human-written questions (one per passage, sampled with seed 42).
Source: <https://rajpurkar.github.io/SQuAD-explorer/>. License: CC BY-SA 4.0, see `data/NOTICE.txt`. Nothing is synthetic.

## Run

```
dotnet test KernelRag.sln
docker compose up --build        # API on :8000, Qdrant; Ollama expected on the host
```

Set `Store__Provider` to `qdrant`, `weaviate`, `elasticsearch` or `memory`.

## Limitations

- No evaluation results exist yet, and the store contract tests have not yet run against real containers.
- Any future numbers come from one shared CI runner with a single-threaded client on 2,067 vectors. They will not be a production benchmark.
- Each SQuAD question is labeled with one passage, so another passage that also answers it counts as a miss.
- The API has no authentication and is a demo only.
- No screenshot exists yet; the eval workflow is written to record one.
