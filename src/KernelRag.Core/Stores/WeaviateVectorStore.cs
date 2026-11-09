using System.Globalization;
using System.Net;
using System.Text.Json.Nodes;

namespace KernelRag.Core.Stores;

public sealed class WeaviateVectorStore(HttpClient http, VectorStoreOptions options) : HttpVectorStore(http, options)
{
    public override string Name => "weaviate";
    private string ClassName => Options.Collection;

    public override async Task EnsureCreatedAsync(int dimensions, CancellationToken ct = default)
    {
        var existing = await SendAsync(HttpMethod.Get, $"v1/schema/{ClassName}", null, ct, HttpStatusCode.NotFound);
        if (existing is null) await CreateAsync(ct);
    }

    public override async Task ResetAsync(int dimensions, CancellationToken ct = default)
    {
        await SendAsync(HttpMethod.Delete, $"v1/schema/{ClassName}", null, ct, HttpStatusCode.NotFound);
        await CreateAsync(ct);
    }

    private Task CreateAsync(CancellationToken ct)
    {
        static JsonObject Property(string name) => new()
        {
            ["name"] = name,
            ["dataType"] = new JsonArray("text"),
            // No keyword indexing: the comparison measures vector indexing only.
            ["indexFilterable"] = false,
            ["indexSearchable"] = false,
        };
        var body = new JsonObject
        {
            ["class"] = ClassName,
            ["vectorizer"] = "none",
            ["vectorIndexType"] = "hnsw",
            ["vectorIndexConfig"] = new JsonObject
            {
                ["distance"] = "cosine",
                ["maxConnections"] = Options.HnswM,
                ["efConstruction"] = Options.EfConstruction,
                ["ef"] = Options.EfSearch,
            },
            ["properties"] = new JsonArray(Property("docId"), Property("title"), Property("text")),
        };
        return SendAsync(HttpMethod.Post, "v1/schema", body, ct);
    }

    public override async Task UpsertAsync(IReadOnlyList<VectorRecord> records, CancellationToken ct = default)
    {
        var objects = new JsonArray();
        foreach (var r in records)
            objects.Add(new JsonObject
            {
                ["class"] = ClassName,
                ["id"] = StableGuid(r.Id).ToString(),
                ["vector"] = VectorNode(r.Vector),
                ["properties"] = new JsonObject { ["docId"] = r.Id, ["title"] = r.Title, ["text"] = r.Text },
            });
        var response = await SendAsync(HttpMethod.Post, "v1/batch/objects", new JsonObject { ["objects"] = objects }, ct);
        // The batch endpoint answers 200 even when individual objects fail.
        var failure = response?.AsArray().FirstOrDefault(o => o?["result"]?["errors"] is not null);
        if (failure is not null)
            throw new VectorStoreException($"weaviate: batch error: {failure["result"]!["errors"]!.ToJsonString()}");
    }

    // HNSW insertion is synchronous in Weaviate's default (non-async-indexing) mode.
    public override Task WaitUntilReadyAsync(CancellationToken ct = default) => Task.CompletedTask;

    public override async Task<IReadOnlyList<SearchHit>> SearchAsync(float[] vector, int k, CancellationToken ct = default)
    {
        var literal = string.Join(",", vector.Select(v => v.ToString("R", CultureInfo.InvariantCulture)));
        var query = $"{{Get{{{ClassName}(nearVector:{{vector:[{literal}]}} limit:{k}){{docId title text _additional{{distance}}}}}}}}";
        return ParseHits(await GraphQlAsync(query, ct), ClassName);
    }

    public static IReadOnlyList<SearchHit> ParseHits(JsonNode? response, string className) =>
        response?["data"]?["Get"]?[className]?.AsArray().Select(o => new SearchHit(
            o!["docId"]!.GetValue<string>(),
            o["title"]!.GetValue<string>(),
            o["text"]!.GetValue<string>(),
            1 - o["_additional"]!["distance"]!.GetValue<double>())).ToList() ?? [];

    public override async Task<long> CountAsync(CancellationToken ct = default)
    {
        var response = await GraphQlAsync($"{{Aggregate{{{ClassName}{{meta{{count}}}}}}}}", ct);
        return response?["data"]?["Aggregate"]?[ClassName]?[0]?["meta"]?["count"]?.GetValue<long>() ?? 0;
    }

    private async Task<JsonNode?> GraphQlAsync(string query, CancellationToken ct)
    {
        var response = await SendAsync(HttpMethod.Post, "v1/graphql", new JsonObject { ["query"] = query }, ct);
        if (response?["errors"] is { } errors)
            throw new VectorStoreException($"weaviate: GraphQL error: {errors.ToJsonString()}");
        return response;
    }
}
