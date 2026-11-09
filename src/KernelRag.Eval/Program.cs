using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using KernelRag.Core;
using KernelRag.Core.Stores;
using KernelRag.Eval;

var command = args.FirstOrDefault();
var opts = Args.Parse(args.Skip(1).ToArray());

switch (command)
{
    case "prepare":
        await SquadData.PrepareAsync(opts.Get("input", SquadData.DevUrl), opts.Get("data", "data"),
            opts.Int("questions", 500), opts.Int("seed", 42));
        return 0;
    case "run":
        return await RunAsync();
    default:
        Console.Error.WriteLine("usage: prepare [--input path|url] [--data dir] | run [--data dir] [--out file] [--gate file] [--groundedness n]");
        return 2;
}

async Task<int> RunAsync()
{
    const int Dimensions = 768;
    const int FetchK = 10;
    const string Collection = "KernelRagSquad";
    var dataDir = opts.Get("data", "data");
    var corpus = SquadData.Load<SquadData.Passage>(Path.Combine(dataDir, "corpus.jsonl"));
    var questions = SquadData.Load<SquadData.Question>(Path.Combine(dataDir, "questions.jsonl"));
    var indexTrials = opts.Int("index-trials", 3);
    var warmup = opts.Int("warmup", 50);
    var groundednessN = opts.Int("groundedness", 0);

    var ollama = new OllamaOptions
    {
        Endpoint = opts.Get("ollama", "http://localhost:11434"),
        ChatModel = opts.Get("chat-model", "qwen2.5:1.5b"),
        EmbeddingModel = "nomic-embed-text",
    };
    var judgeModel = opts.Get("judge-model", "qwen2.5:3b");

    IVectorStore Make(string provider, string url) => VectorStoreFactory.Create(
        new VectorStoreOptions { Provider = provider, Url = url, Collection = Collection });
    var stores = new IVectorStore[]
    {
        Make("qdrant", opts.Get("qdrant", "http://localhost:6333")),
        Make("weaviate", opts.Get("weaviate", "http://localhost:8080")),
        Make("elasticsearch", opts.Get("elasticsearch", "http://localhost:9200")),
        new InMemoryVectorStore(),
    };

    var kernel = KernelFactory.CreateOllama(ollama);
    var rag = new RagService(kernel, stores[^1]);

    // One set of vectors, reused for every store, so only the engine differs.
    Console.WriteLine($"Embedding {corpus.Count} passages and {questions.Count} questions");
    var embedClock = Stopwatch.StartNew();
    var docVectors = await rag.EmbedDocumentsAsync(corpus.Select(p => p.Text).ToList());
    var embedDocsSeconds = embedClock.Elapsed.TotalSeconds;
    var queryVectors = new float[questions.Count][];
    for (var i = 0; i < questions.Count; i++) queryVectors[i] = await rag.EmbedQueryAsync(questions[i].Text);
    if (docVectors[0].Length != Dimensions) throw new InvalidOperationException($"unexpected dimension {docVectors[0].Length}");

    var records = corpus.Select((p, i) => new VectorRecord(p.Id, p.Title, p.Text, docVectors[i])).ToList();
    var indexSeconds = stores.ToDictionary(s => s.Name, _ => new List<double>());
    foreach (var store in stores)
        for (var trial = 0; trial < indexTrials; trial++)
        {
            await store.ResetAsync(Dimensions);
            var clock = Stopwatch.StartNew();
            foreach (var batch in records.Chunk(100)) await store.UpsertAsync(batch);
            await store.WaitUntilReadyAsync();
            indexSeconds[store.Name].Add(clock.Elapsed.TotalSeconds);
            var count = await store.CountAsync();
            if (count != records.Count) throw new InvalidOperationException($"{store.Name}: indexed {count}, expected {records.Count}");
            Console.WriteLine($"{store.Name} trial {trial + 1}: {indexSeconds[store.Name][^1]:F2}s");
        }

    foreach (var store in stores)
        for (var i = 0; i < warmup; i++)
            await store.SearchAsync(queryVectors[i % queryVectors.Length], FetchK);

    // Stores are queried in rotating order per question, so runner drift is spread evenly across them.
    var hits = stores.ToDictionary(s => s.Name, _ => new IReadOnlyList<SearchHit>[questions.Count]);
    var latencies = stores.ToDictionary(s => s.Name, _ => new List<double>(questions.Count));
    for (var q = 0; q < questions.Count; q++)
        for (var offset = 0; offset < stores.Length; offset++)
        {
            var store = stores[(q + offset) % stores.Length];
            var clock = Stopwatch.StartNew();
            var result = await store.SearchAsync(queryVectors[q], FetchK);
            latencies[store.Name].Add(clock.Elapsed.TotalMilliseconds);
            hits[store.Name][q] = result;
        }

    var results = new JsonObject();
    foreach (var store in stores)
    {
        var ranked = hits[store.Name].Select(h => (IReadOnlyList<string>)h.Select(x => x.Id).ToList()).ToList();
        int Hits(int k) => ranked.Select((r, q) => Metrics.HitAt(r, questions[q].PassageId, k)).Sum();
        var (low, high) = Metrics.Wilson(Hits(5), questions.Count);
        var trials = indexSeconds[store.Name];
        results[store.Name] = new JsonObject
        {
            ["recall_at_1"] = R((double)Hits(1) / questions.Count),
            ["recall_at_3"] = R((double)Hits(3) / questions.Count),
            ["recall_at_5"] = R((double)Hits(5) / questions.Count),
            ["recall_at_5_ci95"] = new JsonArray(R(low), R(high)),
            ["recall_at_10"] = R((double)Hits(10) / questions.Count),
            ["mrr"] = R(ranked.Select((r, q) => Metrics.ReciprocalRank(r, questions[q].PassageId)).Average()),
            ["indexing_seconds_median"] = R(Metrics.Percentile(trials, 50)),
            ["indexing_seconds_trials"] = new JsonArray(trials.Select(t => (JsonNode)R(t)).ToArray()),
            ["query_latency_ms_p50"] = R(Metrics.Percentile(latencies[store.Name], 50)),
            ["query_latency_ms_p95"] = R(Metrics.Percentile(latencies[store.Name], 95)),
            ["in_process_baseline"] = store is InMemoryVectorStore,
        };
        Console.WriteLine($"{store.Name}: {results[store.Name]!.ToJsonString()}");
    }

    JsonObject? groundedness = null;
    if (groundednessN > 0)
    {
        var judge = KernelFactory.CreateOllama(new OllamaOptions
        {
            Endpoint = ollama.Endpoint, ChatModel = judgeModel, EmbeddingModel = ollama.EmbeddingModel,
        });
        groundedness = await GroundednessEval.RunAsync(rag, judge, questions, hits,
            stores.Where(s => s is not InMemoryVectorStore).Select(s => s.Name).ToList(), groundednessN);
    }

    var output = new JsonObject
    {
        ["generated_at_utc"] = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
        ["conditions"] = new JsonObject
        {
            ["runner"] = Environment.GetEnvironmentVariable("RUNNER_NAME") ?? Environment.MachineName,
            ["runner_os"] = Environment.OSVersion.ToString(),
            ["logical_cpus"] = Environment.ProcessorCount,
            ["total_memory_gb"] = R(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 1073741824.0),
            ["commit"] = Environment.GetEnvironmentVariable("GITHUB_SHA") ?? "local",
            ["github_run_id"] = Environment.GetEnvironmentVariable("GITHUB_RUN_ID") ?? "local",
            ["images"] = new JsonObject
            {
                ["qdrant"] = Environment.GetEnvironmentVariable("QDRANT_IMAGE") ?? "unknown",
                ["weaviate"] = Environment.GetEnvironmentVariable("WEAVIATE_IMAGE") ?? "unknown",
                ["elasticsearch"] = Environment.GetEnvironmentVariable("ELASTICSEARCH_IMAGE") ?? "unknown",
            },
            ["dataset"] = "SQuAD 1.1 dev (human-written questions; CC BY-SA 4.0)",
            ["passages"] = corpus.Count,
            ["questions"] = questions.Count,
            ["embedding_model"] = ollama.EmbeddingModel,
            ["embedding_dimensions"] = Dimensions,
            ["embedding_seconds_for_passages"] = R(embedDocsSeconds),
            ["hnsw"] = new JsonObject { ["m"] = 16, ["ef_construction"] = 100, ["ef_search"] = 100 },
            ["k_fetched"] = FetchK,
            ["warmup_queries_excluded"] = warmup,
            ["index_trials"] = indexTrials,
            ["client"] = "single-threaded sequential HTTP from one .NET process on the same host as the containers",
        },
        ["stores"] = results,
        ["groundedness"] = groundedness,
    };

    var outPath = opts.Get("out", "evals/results/latest.json");
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
    File.WriteAllText(outPath, output.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"Wrote {outPath}");

    var gatePath = opts.Get("gate", "");
    if (gatePath.Length == 0) return 0;
    var failures = Gate.Check(JsonNode.Parse(File.ReadAllText(gatePath))!, results);
    foreach (var failure in failures) Console.Error.WriteLine($"GATE FAILED: {failure}");
    if (failures.Count == 0) Console.WriteLine("Gate passed");
    return failures.Count == 0 ? 0 : 1;
}

static JsonNode R(double value) => JsonValue.Create(Math.Round(value, 4))!;
