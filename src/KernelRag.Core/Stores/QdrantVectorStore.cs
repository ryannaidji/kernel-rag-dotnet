using System.Net;
using System.Text.Json.Nodes;

namespace KernelRag.Core.Stores;

public sealed class QdrantVectorStore(HttpClient http, VectorStoreOptions options) : HttpVectorStore(http, options)
{
    private const int IndexingThresholdKb = 10;

    public override string Name => "qdrant";
    private string Path => $"collections/{Options.Collection}";

    public override async Task EnsureCreatedAsync(int dimensions, CancellationToken ct = default)
    {
        var existing = await SendAsync(HttpMethod.Get, Path, null, ct, HttpStatusCode.NotFound);
        if (existing is null) await CreateAsync(dimensions, ct);
    }

    public override async Task ResetAsync(int dimensions, CancellationToken ct = default)
    {
        await SendAsync(HttpMethod.Delete, Path, null, ct, HttpStatusCode.NotFound);
        await CreateAsync(dimensions, ct);
    }

    private Task CreateAsync(int dimensions, CancellationToken ct)
    {
        // Qdrant's defaults would leave a collection this small unindexed and answer by exact
        // scan (indexing_threshold 20000 KB, full_scan_threshold 10000 KB). Both are lowered so
        // this store really runs HNSW, like the other two.
        var body = new JsonObject
        {
            ["vectors"] = new JsonObject { ["size"] = dimensions, ["distance"] = "Cosine" },
            ["hnsw_config"] = new JsonObject
            {
                ["m"] = Options.HnswM,
                ["ef_construct"] = Options.EfConstruction,
                ["full_scan_threshold"] = 10,
            },
            ["optimizers_config"] = new JsonObject
            {
                ["indexing_threshold"] = IndexingThresholdKb,
                ["default_segment_number"] = 1,
            },
        };
        return SendAsync(HttpMethod.Put, Path, body, ct);
    }

    public override Task UpsertAsync(IReadOnlyList<VectorRecord> records, CancellationToken ct = default)
    {
        var points = new JsonArray();
        foreach (var r in records)
            points.Add(new JsonObject
            {
                ["id"] = StableGuid(r.Id).ToString(),
                ["vector"] = VectorNode(r.Vector),
                ["payload"] = new JsonObject { ["doc_id"] = r.Id, ["title"] = r.Title, ["text"] = r.Text },
            });
        return SendAsync(HttpMethod.Put, $"{Path}/points?wait=true", new JsonObject { ["points"] = points }, ct);
    }

    public override Task WaitUntilReadyAsync(CancellationToken ct = default) =>
        PollUntilAsync(async () =>
        {
            var result = (await SendAsync(HttpMethod.Get, Path, null, ct))?["result"];
            var points = result?["points_count"]?.GetValue<long>() ?? 0;
            var indexed = result?["indexed_vectors_count"]?.GetValue<long>() ?? 0;
            var size = result?["config"]?["params"]?["vectors"]?["size"]?.GetValue<long>() ?? 0;
            // A collection under indexing_threshold is never indexed, so there is nothing to wait for.
            var belowThreshold = points * size * sizeof(float) < IndexingThresholdKb * 1024;
            return result?["status"]?.GetValue<string>() == "green" && (indexed >= points || belowThreshold);
        }, "the HNSW index to finish building", ct);

    public override async Task<IReadOnlyList<SearchHit>> SearchAsync(float[] vector, int k, CancellationToken ct = default)
    {
        var body = new JsonObject
        {
            ["query"] = VectorNode(vector),
            ["limit"] = k,
            ["with_payload"] = true,
            ["params"] = new JsonObject { ["hnsw_ef"] = Math.Max(Options.EfSearch, k) },
        };
        return ParseHits(await SendAsync(HttpMethod.Post, $"{Path}/points/query", body, ct));
    }

    public static IReadOnlyList<SearchHit> ParseHits(JsonNode? response) =>
        response?["result"]?["points"]?.AsArray().Select(p => new SearchHit(
            p!["payload"]!["doc_id"]!.GetValue<string>(),
            p["payload"]!["title"]!.GetValue<string>(),
            p["payload"]!["text"]!.GetValue<string>(),
            p["score"]!.GetValue<double>())).ToList() ?? [];

    public override async Task<long> CountAsync(CancellationToken ct = default)
    {
        var response = await SendAsync(HttpMethod.Post, $"{Path}/points/count", new JsonObject { ["exact"] = true }, ct);
        return response?["result"]?["count"]?.GetValue<long>() ?? 0;
    }
}
