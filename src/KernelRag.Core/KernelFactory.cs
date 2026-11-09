using Microsoft.SemanticKernel;

namespace KernelRag.Core;

public static class KernelFactory
{
    /// <summary>Builds a Semantic Kernel wired to local Ollama for chat and embeddings.</summary>
    public static Kernel CreateOllama(OllamaOptions options)
    {
        // CPU inference can exceed HttpClient's default 100 s timeout.
        HttpClient Client() => new() { BaseAddress = new Uri(options.Endpoint), Timeout = TimeSpan.FromMinutes(15) };

        var builder = Kernel.CreateBuilder();
        builder.AddOllamaChatCompletion(options.ChatModel, Client());
        builder.AddOllamaEmbeddingGenerator(options.EmbeddingModel, Client());
        return builder.Build();
    }
}
