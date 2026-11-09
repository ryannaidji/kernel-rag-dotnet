using KernelRag.Core;
using KernelRag.Core.Stores;
using Xunit;

namespace KernelRag.Tests;

/// <summary>Runs only when RUN_STORE_INTEGRATION=1 (CI starts the three real containers).</summary>
public sealed class StoreFactAttribute : TheoryAttribute
{
    public StoreFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("RUN_STORE_INTEGRATION") != "1")
            Skip = "Set RUN_STORE_INTEGRATION=1 with the store containers running";
    }
}

public class StoreIntegrationTests
{
    private const int Dim = FakeEmbedder.Dimensions;

    private static IVectorStore Make(string provider, string url, string collection) =>
        VectorStoreFactory.Create(new VectorStoreOptions { Provider = provider, Url = url, Collection = collection });

    private static VectorRecord Doc(string id, string text) => new(id, "Title " + id, text, FakeEmbedder.Embed(text));

    [StoreFact]
    [InlineData("qdrant", "http://localhost:6333")]
    [InlineData("weaviate", "http://localhost:8080")]
    [InlineData("elasticsearch", "http://localhost:9200")]
    public async Task RoundTrip_ContractHolds(string provider, string url)
    {
        var store = Make(provider, url, "ContractTest");
        await store.ResetAsync(Dim);
        await store.UpsertAsync([
            Doc("paris", "the capital of france is paris"),
            Doc("tokyo", "sushi in tokyo japan"),
            Doc("rain", "rain forest biology of frogs"),
        ]);
        await store.WaitUntilReadyAsync();

        Assert.Equal(3, await store.CountAsync());
        var hits = await store.SearchAsync(FakeEmbedder.Embed("capital of france"), 2);

        Assert.Equal(2, hits.Count);
        Assert.Equal("paris", hits[0].Id);
        Assert.Equal("Title paris", hits[0].Title);
        Assert.Equal("the capital of france is paris", hits[0].Text);
        Assert.True(hits[0].Score >= hits[1].Score);

        var expected = InMemoryVectorStore.Cosine(FakeEmbedder.Embed("capital of france"), FakeEmbedder.Embed("the capital of france is paris"));
        Assert.Equal(expected, hits[0].Score, 2);
    }

    [StoreFact]
    [InlineData("qdrant", "http://localhost:6333")]
    [InlineData("weaviate", "http://localhost:8080")]
    [InlineData("elasticsearch", "http://localhost:9200")]
    public async Task Upsert_ReplacesExistingId(string provider, string url)
    {
        var store = Make(provider, url, "UpsertTest");
        await store.ResetAsync(Dim);
        await store.UpsertAsync([Doc("a", "old text about cats")]);
        await store.UpsertAsync([Doc("a", "new text about dogs")]);
        await store.WaitUntilReadyAsync();

        Assert.Equal(1, await store.CountAsync());
        var hits = await store.SearchAsync(FakeEmbedder.Embed("dogs"), 1);
        Assert.Equal("new text about dogs", hits.Single().Text);
    }
}
