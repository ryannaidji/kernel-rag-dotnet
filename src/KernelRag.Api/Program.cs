using KernelRag.Core;
using KernelRag.Core.Stores;
using Microsoft.SemanticKernel;

var builder = WebApplication.CreateBuilder(args);

var storeOptions = builder.Configuration.GetSection("Store").Get<VectorStoreOptions>() ?? new();
var ollamaOptions = builder.Configuration.GetSection("Ollama").Get<OllamaOptions>() ?? new();

builder.Services.AddSingleton<IVectorStore>(_ => VectorStoreFactory.Create(storeOptions));
builder.Services.AddSingleton<Kernel>(_ => KernelFactory.CreateOllama(ollamaOptions));
builder.Services.AddSingleton<RagService>(sp => new RagService(sp.GetRequiredService<Kernel>(), sp.GetRequiredService<IVectorStore>()));

var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/api/health", (IVectorStore store) => Results.Ok(new { status = "ok", store = store.Name }));

app.MapPost("/api/ingest", async (IngestRequest request, RagService rag, CancellationToken ct) =>
{
    if (request.Documents is not { Count: > 0 }) return Results.BadRequest(new { error = "documents is required" });
    var chunks = await rag.IngestAsync(request.Documents, ct);
    return Results.Ok(new { chunks });
});

app.MapPost("/api/ask", async (AskRequest request, RagService rag, CancellationToken ct) =>
{
    if (string.IsNullOrWhiteSpace(request.Question)) return Results.BadRequest(new { error = "question is required" });
    if (request.K is < 1 or > 20) return Results.BadRequest(new { error = "k must be between 1 and 20" });
    var result = await rag.AskAsync(request.Question, request.K, ct);
    return Results.Ok(new { answer = result.Answer, sources = result.Sources });
});

app.Run();

public sealed record IngestRequest(List<SourceDocument> Documents);
public sealed record AskRequest(string Question, int K = 5);

public partial class Program;
