namespace KernelRag.Core.Stores;

/// <summary>The single seam between the RAG pipeline and a vector database.</summary>
public interface IVectorStore
{
    string Name { get; }

    /// <summary>Creates the collection if it does not exist.</summary>
    Task EnsureCreatedAsync(int dimensions, CancellationToken ct = default);

    /// <summary>Drops the collection, then creates it empty.</summary>
    Task ResetAsync(int dimensions, CancellationToken ct = default);

    Task UpsertAsync(IReadOnlyList<VectorRecord> records, CancellationToken ct = default);

    /// <summary>Returns once written data is searchable through the vector index.</summary>
    Task WaitUntilReadyAsync(CancellationToken ct = default);

    Task<IReadOnlyList<SearchHit>> SearchAsync(float[] vector, int k, CancellationToken ct = default);

    Task<long> CountAsync(CancellationToken ct = default);
}
