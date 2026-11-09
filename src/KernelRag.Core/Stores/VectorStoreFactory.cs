namespace KernelRag.Core.Stores;

public static class VectorStoreFactory
{
    public static IVectorStore Create(VectorStoreOptions options)
    {
        if (options.Provider.Equals("memory", StringComparison.OrdinalIgnoreCase))
            return new InMemoryVectorStore();

        var http = new HttpClient
        {
            BaseAddress = new Uri(options.Url.TrimEnd('/') + "/"),
            Timeout = TimeSpan.FromMinutes(5),
        };
        return options.Provider.ToLowerInvariant() switch
        {
            "qdrant" => new QdrantVectorStore(http, options),
            "weaviate" => new WeaviateVectorStore(http, options),
            "elasticsearch" => new ElasticsearchVectorStore(http, options),
            _ => throw new ArgumentException($"Unknown vector store provider '{options.Provider}'"),
        };
    }
}
