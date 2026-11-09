using System.Net;
using System.Text;
using KernelRag.Core;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace KernelRag.Tests;

/// <summary>Deterministic bag-of-words embedding: texts that share words have similar vectors.</summary>
public sealed class FakeEmbedder : IEmbeddingGenerator<string, Embedding<float>>
{
    public const int Dimensions = 64;

    public static float[] Embed(string text)
    {
        var vector = new float[Dimensions];
        foreach (var token in Groundedness.Tokens(text.Replace(RagService.QueryPrefix, "").Replace(RagService.DocumentPrefix, "")))
        {
            var hash = 17;
            foreach (var c in token) hash = unchecked(hash * 31 + c);
            vector[Math.Abs(hash % Dimensions)] += 1;
        }
        return vector;
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(values.Select(v => new Embedding<float>(Embed(v)))));

    public object? GetService(Type serviceType, object? serviceKey = null) => null;
    public void Dispose() { }
}

public sealed class FakeChat(string reply) : IChatCompletionService
{
    public ChatHistory? LastHistory { get; private set; }
    public IReadOnlyDictionary<string, object?> Attributes { get; } = new Dictionary<string, object?>();

    public Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        LastHistory = chatHistory;
        IReadOnlyList<ChatMessageContent> result = [new(AuthorRole.Assistant, reply)];
        return Task.FromResult(result);
    }

    public IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null, Kernel? kernel = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

public static class TestKernel
{
    public static Kernel Create(IChatCompletionService chat)
    {
        var builder = Kernel.CreateBuilder();
        builder.Services.AddSingleton(chat);
        builder.Services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(new FakeEmbedder());
        return builder.Build();
    }
}

/// <summary>Captures requests and returns a canned response.</summary>
public sealed class CapturingHandler(HttpStatusCode status, string response) : HttpMessageHandler
{
    public List<(HttpMethod Method, string Path, string Body)> Requests { get; } = [];

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        var body = request.Content is null ? "" : await request.Content.ReadAsStringAsync(ct);
        Requests.Add((request.Method, request.RequestUri!.AbsolutePath, body));
        return new HttpResponseMessage(status) { Content = new StringContent(response, Encoding.UTF8, "application/json") };
    }
}
