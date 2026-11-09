using System.Text;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Ollama;
using KernelRag.Core.Stores;

namespace KernelRag.Core;

public sealed class RagService(Kernel kernel, IVectorStore store)
{
    // nomic-embed-text is trained with these task prefixes.
    public const string QueryPrefix = "search_query: ";
    public const string DocumentPrefix = "search_document: ";

    // Message tags let Semantic Kernel split roles; variables are XML-encoded when rendered and
    // decoded by the prompt parser, so retrieved text cannot inject its own message tags.
    private const string AnswerPrompt = """
        <message role="system">Answer the question using only the context. Answer briefly. If the context does not contain the answer, reply exactly: I don't know.</message>
        <message role="user">Context:
        {{$context}}

        Question: {{$question}}</message>
        """;

    private readonly IEmbeddingGenerator<string, Embedding<float>> _embedder =
        kernel.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

    public async Task<int> IngestAsync(IEnumerable<SourceDocument> documents, CancellationToken ct = default)
    {
        var chunks = documents.SelectMany(d => Chunker.Chunk(d.Text)
            .Select((text, i) => new SourceDocument($"{d.Id}#{i}", d.Title, text))).ToList();
        if (chunks.Count == 0) return 0;

        var vectors = await EmbedDocumentsAsync(chunks.Select(c => c.Text).ToList(), ct);
        await store.EnsureCreatedAsync(vectors[0].Length, ct);
        await store.UpsertAsync(chunks.Select((c, i) => new VectorRecord(c.Id, c.Title, c.Text, vectors[i])).ToList(), ct);
        await store.WaitUntilReadyAsync(ct);
        return chunks.Count;
    }

    public async Task<float[][]> EmbedDocumentsAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        var result = new List<float[]>(texts.Count);
        foreach (var batch in texts.Chunk(16))
        {
            var embeddings = await _embedder.GenerateAsync(batch.Select(t => DocumentPrefix + t), cancellationToken: ct);
            result.AddRange(embeddings.Select(e => e.Vector.ToArray()));
        }
        return result.ToArray();
    }

    public async Task<float[]> EmbedQueryAsync(string query, CancellationToken ct = default)
    {
        var embeddings = await _embedder.GenerateAsync([QueryPrefix + query], cancellationToken: ct);
        return embeddings[0].Vector.ToArray();
    }

    public async Task<RagAnswer> AskAsync(string question, int k = 5, CancellationToken ct = default)
    {
        var hits = await store.SearchAsync(await EmbedQueryAsync(question, ct), k, ct);
        return new RagAnswer(await GenerateAsync(question, hits, ct), hits);
    }

    public async Task<string> GenerateAsync(string question, IReadOnlyList<SearchHit> hits, CancellationToken ct = default)
    {
        var settings = new OllamaPromptExecutionSettings { Temperature = 0, NumPredict = 128 };
        var function = kernel.CreateFunctionFromPrompt(AnswerPrompt, settings);
        var result = await kernel.InvokeAsync(function, new KernelArguments(settings)
        {
            ["context"] = FormatContext(hits),
            ["question"] = question,
        }, ct);
        return result.ToString().Trim();
    }

    public static string FormatContext(IEnumerable<SearchHit> hits)
    {
        var sb = new StringBuilder();
        foreach (var h in hits) sb.Append('[').Append(h.Title).Append("] ").Append(h.Text).Append("\n\n");
        return sb.ToString().TrimEnd();
    }
}
