using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace KernelRag.Core.Stores;

/// <summary>
/// Shared plumbing. All three engines are called over their plain REST APIs through this one
/// HttpClient path, so client-library differences do not leak into the latency comparison.
/// </summary>
public abstract class HttpVectorStore(HttpClient http, VectorStoreOptions options) : IVectorStore
{
    protected HttpClient Http { get; } = http;
    protected VectorStoreOptions Options { get; } = options;

    public abstract string Name { get; }
    public abstract Task EnsureCreatedAsync(int dimensions, CancellationToken ct = default);
    public abstract Task ResetAsync(int dimensions, CancellationToken ct = default);
    public abstract Task UpsertAsync(IReadOnlyList<VectorRecord> records, CancellationToken ct = default);
    public abstract Task WaitUntilReadyAsync(CancellationToken ct = default);
    public abstract Task<IReadOnlyList<SearchHit>> SearchAsync(float[] vector, int k, CancellationToken ct = default);
    public abstract Task<long> CountAsync(CancellationToken ct = default);

    protected Task<JsonNode?> SendAsync(HttpMethod method, string path, JsonNode? body, CancellationToken ct,
        params HttpStatusCode[] alsoAllowed) =>
        SendRawAsync(method, path, body?.ToJsonString(), "application/json", ct, alsoAllowed);

    protected async Task<JsonNode?> SendRawAsync(HttpMethod method, string path, string? content, string contentType,
        CancellationToken ct, params HttpStatusCode[] alsoAllowed)
    {
        using var request = new HttpRequestMessage(method, path);
        if (content is not null)
        {
            request.Content = new StringContent(content, Encoding.UTF8);
            request.Content.Headers.ContentType = new(contentType);
        }
        using var response = await Http.SendAsync(request, ct);
        var text = await response.Content.ReadAsStringAsync(ct);
        if (!response.IsSuccessStatusCode && !alsoAllowed.Contains(response.StatusCode))
            throw new VectorStoreException(
                $"{Name}: {method} {path} returned {(int)response.StatusCode}: {Truncate(text)}");
        if (!response.IsSuccessStatusCode || string.IsNullOrWhiteSpace(text)) return null;
        return JsonNode.Parse(text);
    }

    protected static JsonNode VectorNode(float[] vector) => JsonSerializer.SerializeToNode(vector)!;

    /// <summary>Qdrant and Weaviate need UUID ids; derive a stable one from the document id.</summary>
    public static Guid StableGuid(string id) => new(MD5.HashData(Encoding.UTF8.GetBytes(id)));

    protected async Task PollUntilAsync(Func<Task<bool>> ready, string what, CancellationToken ct)
    {
        var deadline = DateTime.UtcNow.AddSeconds(Options.ReadyTimeoutSeconds);
        while (!await ready())
        {
            if (DateTime.UtcNow > deadline)
                throw new VectorStoreException($"{Name}: timed out waiting for {what}");
            await Task.Delay(100, ct);
        }
    }

    private static string Truncate(string s) => s.Length <= 500 ? s : s[..500];
}

public sealed class VectorStoreException(string message) : Exception(message);
