using System.Net;
using System.Text;
using System.Text.Json.Nodes;

namespace KernelRag.Core.Stores;

public sealed class ElasticsearchVectorStore(HttpClient http, VectorStoreOptions options) : HttpVectorStore(http, options)
{
    public override string Name => "elasticsearch";
    private string Index => Options.Collection.ToLowerInvariant();

    public override async Task EnsureCreatedAsync(int dimensions, CancellationToken ct = default)
    {
        var existing = await SendAsync(HttpMethod.Get, Index, null, ct, HttpStatusCode.NotFound);
        if (existing is null) await CreateAsync(dimensions, ct);
    }

    public override async Task ResetAsync(int dimensions, CancellationToken ct = default)
    {
        await SendAsync(HttpMethod.Delete, Index, null, ct, HttpStatusCode.NotFound);
        await CreateAsync(dimensions, ct);
    }

    private Task CreateAsync(int dimensions, CancellationToken ct)
    {
        static JsonObject NotIndexed(string type) => new() { ["type"] = type, ["index"] = false };
        var body = new JsonObject
        {
            ["settings"] = new JsonObject { ["number_of_shards"] = 1, ["number_of_replicas"] = 0 },
            ["mappings"] = new JsonObject
            {
                ["properties"] = new JsonObject
                {
                    ["doc_id"] = NotIndexed("keyword"),
                    ["title"] = NotIndexed("text"),
                    ["text"] = NotIndexed("text"),
                    ["embedding"] = new JsonObject
                    {
                        ["type"] = "dense_vector",
                        ["dims"] = dimensions,
                        ["index"] = true,
                        ["similarity"] = "cosine",
                        // Plain float HNSW: the 8.x default is int8 quantization, which would not be like for like.
                        ["index_options"] = new JsonObject
                        {
                            ["type"] = "hnsw",
                            ["m"] = Options.HnswM,
                            ["ef_construction"] = Options.EfConstruction,
                        },
                    },
                },
            },
        };
        return SendAsync(HttpMethod.Put, Index, body, ct);
    }

    public override async Task UpsertAsync(IReadOnlyList<VectorRecord> records, CancellationToken ct = default)
    {
        var ndjson = new StringBuilder();
        foreach (var r in records)
        {
            ndjson.Append(new JsonObject { ["index"] = new JsonObject { ["_id"] = r.Id } }.ToJsonString()).Append('\n');
            ndjson.Append(new JsonObject
            {
                ["doc_id"] = r.Id,
                ["title"] = r.Title,
                ["text"] = r.Text,
                ["embedding"] = VectorNode(r.Vector),
            }.ToJsonString()).Append('\n');
        }
        var response = await SendRawAsync(HttpMethod.Post, $"{Index}/_bulk", ndjson.ToString(), "application/x-ndjson", ct);
        if (response?["errors"]?.GetValue<bool>() == true)
        {
            var failure = response["items"]!.AsArray().First(i => i?["index"]?["error"] is not null);
            throw new VectorStoreException($"elasticsearch: bulk error: {failure!["index"]!["error"]!.ToJsonString()}");
        }
    }

    public override Task WaitUntilReadyAsync(CancellationToken ct = default) =>
        SendAsync(HttpMethod.Post, $"{Index}/_refresh", null, ct);

    public override async Task<IReadOnlyList<SearchHit>> SearchAsync(float[] vector, int k, CancellationToken ct = default)
    {
        var body = new JsonObject
        {
            ["size"] = k,
            ["_source"] = new JsonArray("doc_id", "title", "text"),
            ["knn"] = new JsonObject
            {
                ["field"] = "embedding",
                ["query_vector"] = VectorNode(vector),
                ["k"] = k,
                ["num_candidates"] = Math.Max(Options.EfSearch, k),
            },
        };
        return ParseHits(await SendAsync(HttpMethod.Post, $"{Index}/_search", body, ct));
    }

    // Elasticsearch reports cosine as (1 + cos) / 2; convert back to plain cosine.
    public static IReadOnlyList<SearchHit> ParseHits(JsonNode? response) =>
        response?["hits"]?["hits"]?.AsArray().Select(h => new SearchHit(
            h!["_source"]!["doc_id"]!.GetValue<string>(),
            h["_source"]!["title"]!.GetValue<string>(),
            h["_source"]!["text"]!.GetValue<string>(),
            2 * h["_score"]!.GetValue<double>() - 1)).ToList() ?? [];

    public override async Task<long> CountAsync(CancellationToken ct = default)
    {
        var response = await SendAsync(HttpMethod.Get, $"{Index}/_count", null, ct);
        return response?["count"]?.GetValue<long>() ?? 0;
    }
}
