namespace KernelRag.Core;

public sealed record SourceDocument(string Id, string Title, string Text);
public sealed record VectorRecord(string Id, string Title, string Text, float[] Vector);

/// <summary>A search result. Score is cosine similarity in [-1, 1] for every store.</summary>
public sealed record SearchHit(string Id, string Title, string Text, double Score);

public sealed record RagAnswer(string Answer, IReadOnlyList<SearchHit> Sources);

public sealed class VectorStoreOptions
{
    public string Provider { get; set; } = "qdrant";
    public string Url { get; set; } = "http://localhost:6333";
    public string Collection { get; set; } = "KernelRagDocs";

    // The same HNSW settings are applied to every engine so the comparison is like for like.
    public int HnswM { get; set; } = 16;
    public int EfConstruction { get; set; } = 100;
    public int EfSearch { get; set; } = 100;
    public int ReadyTimeoutSeconds { get; set; } = 300;
}

public sealed class OllamaOptions
{
    public string Endpoint { get; set; } = "http://localhost:11434";
    public string ChatModel { get; set; } = "qwen2.5:1.5b";
    public string EmbeddingModel { get; set; } = "nomic-embed-text";
}
