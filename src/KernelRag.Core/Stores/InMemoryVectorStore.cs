using System.Collections.Concurrent;

namespace KernelRag.Core.Stores;

/// <summary>Exact brute-force search. Used by tests and as the exact-search baseline in the eval.</summary>
public sealed class InMemoryVectorStore : IVectorStore
{
    private readonly ConcurrentDictionary<string, VectorRecord> _records = new();

    public string Name => "exact-in-memory";
    public Task EnsureCreatedAsync(int dimensions, CancellationToken ct = default) => Task.CompletedTask;

    public Task ResetAsync(int dimensions, CancellationToken ct = default)
    {
        _records.Clear();
        return Task.CompletedTask;
    }

    public Task UpsertAsync(IReadOnlyList<VectorRecord> records, CancellationToken ct = default)
    {
        foreach (var r in records) _records[r.Id] = r;
        return Task.CompletedTask;
    }

    public Task WaitUntilReadyAsync(CancellationToken ct = default) => Task.CompletedTask;

    public Task<IReadOnlyList<SearchHit>> SearchAsync(float[] vector, int k, CancellationToken ct = default)
    {
        IReadOnlyList<SearchHit> hits = _records.Values
            .Select(r => new SearchHit(r.Id, r.Title, r.Text, Cosine(vector, r.Vector)))
            .OrderByDescending(h => h.Score).ThenBy(h => h.Id, StringComparer.Ordinal)
            .Take(k).ToList();
        return Task.FromResult(hits);
    }

    public Task<long> CountAsync(CancellationToken ct = default) => Task.FromResult((long)_records.Count);

    public static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return na == 0 || nb == 0 ? 0 : dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }
}
