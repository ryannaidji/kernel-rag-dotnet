using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using KernelRag.Core;
using Microsoft.SemanticKernel;

namespace KernelRag.Eval;

public sealed class Args(Dictionary<string, string> values)
{
    public static Args Parse(string[] args)
    {
        var values = new Dictionary<string, string>();
        for (var i = 0; i + 1 < args.Length; i += 2)
            values[args[i].TrimStart('-')] = args[i + 1];
        return new Args(values);
    }

    public string Get(string name, string fallback) => values.GetValueOrDefault(name, fallback);
    public int Int(string name, int fallback) => values.TryGetValue(name, out var v) ? int.Parse(v) : fallback;
}

public static class Gate
{
    public static List<string> Check(JsonNode gate, JsonObject results)
    {
        var failures = new List<string>();
        foreach (var (store, minimum) in gate["min_recall_at_5"]!.AsObject())
        {
            var actual = results[store]?["recall_at_5"]?.GetValue<double>();
            if (actual is null) failures.Add($"{store}: no result");
            else if (actual < minimum!.GetValue<double>())
                failures.Add($"{store}: recall@5 {actual:F4} is below {minimum.GetValue<double>():F4}");
        }
        return failures;
    }
}

public static class SquadData
{
    public const string DevUrl = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json";

    public sealed record Passage(string Id, string Title, string Text);
    public sealed record Question(string Id, string Text, string PassageId, List<string> Answers);

    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };

    public static async Task PrepareAsync(string input, string dataDir, int questionCount, int seed)
    {
        var raw = input.StartsWith("http", StringComparison.Ordinal)
            ? await new HttpClient().GetStringAsync(input)
            : await File.ReadAllTextAsync(input);
        var (passages, questions) = Parse(raw, questionCount, seed);
        Directory.CreateDirectory(dataDir);
        Save(Path.Combine(dataDir, "corpus.jsonl"), passages);
        Save(Path.Combine(dataDir, "questions.jsonl"), questions);
        Console.WriteLine($"{passages.Count} passages, {questions.Count} questions");
    }

    /// <summary>Keeps every passage and samples at most one answerable question per passage, with a fixed seed.</summary>
    public static (List<Passage>, List<Question>) Parse(string squadJson, int questionCount, int seed)
    {
        var passages = new List<Passage>();
        var candidates = new List<Question>();
        var articles = JsonNode.Parse(squadJson)!["data"]!.AsArray();
        for (var a = 0; a < articles.Count; a++)
        {
            var title = articles[a]!["title"]!.GetValue<string>().Replace('_', ' ');
            var paragraphs = articles[a]!["paragraphs"]!.AsArray();
            for (var p = 0; p < paragraphs.Count; p++)
            {
                var id = $"squad-{a:D2}-{p:D3}";
                passages.Add(new Passage(id, title, paragraphs[p]!["context"]!.GetValue<string>()));
                var first = paragraphs[p]!["qas"]!.AsArray().FirstOrDefault();
                if (first is null) continue;
                candidates.Add(new Question(first["id"]!.GetValue<string>(), first["question"]!.GetValue<string>(), id,
                    first["answers"]!.AsArray().Select(x => x!["text"]!.GetValue<string>()).Distinct().ToList()));
            }
        }
        var random = new Random(seed);
        return (passages, candidates.OrderBy(_ => random.Next()).Take(questionCount).ToList());
    }

    private static void Save<T>(string path, IEnumerable<T> rows) =>
        File.WriteAllLines(path, rows.Select(r => JsonSerializer.Serialize(r, Json)), new UTF8Encoding(false));

    public static List<T> Load<T>(string path) =>
        File.ReadLines(path).Where(l => l.Length > 0).Select(l => JsonSerializer.Deserialize<T>(l, Json)!).ToList();
}

public static class GroundednessEval
{
    public const int ContextSize = 5;

    public static async Task<JsonObject> RunAsync(RagService rag, Kernel judge, List<SquadData.Question> questions,
        Dictionary<string, IReadOnlyList<SearchHit>[]> hits, List<string> storeNames, int sample)
    {
        sample = Math.Min(sample, questions.Count);
        // Identical retrieved context gives an identical temperature-0 answer, so cache by context.
        var answers = new Dictionary<string, string>();
        var verdicts = new Dictionary<string, string>();
        var perStore = new JsonObject();
        var examples = new JsonArray();

        foreach (var name in storeNames)
        {
            int answered = 0, abstained = 0, supported = 0, unparsable = 0, goldFound = 0;
            var lexical = new List<double>();
            for (var q = 0; q < sample; q++)
            {
                var top = hits[name][q].Take(ContextSize).ToList();
                var context = RagService.FormatContext(top);
                var key = q + "|" + string.Join(",", top.Select(h => h.Id));
                if (!answers.TryGetValue(key, out var answer))
                    answers[key] = answer = await rag.GenerateAsync(questions[q].Text, top);

                if (Groundedness.ContainsGold(answer, questions[q].Answers)) goldFound++;
                if (Groundedness.IsAbstention(answer)) { abstained++; continue; }

                answered++;
                lexical.Add(Groundedness.LexicalSupport(answer, context));
                if (!verdicts.TryGetValue(key, out var reply))
                    verdicts[key] = reply = await Groundedness.JudgeAsync(judge, questions[q].Text, answer, context);
                var verdict = Groundedness.ParseVerdict(reply);
                if (verdict is null) unparsable++;
                else if (verdict.Value) supported++;

                if (examples.Count < 5)
                    examples.Add(new JsonObject
                    {
                        ["store"] = name, ["question"] = questions[q].Text, ["answer"] = answer,
                        ["gold"] = questions[q].Answers[0], ["judge_reply"] = reply,
                    });
            }
            perStore[name] = new JsonObject
            {
                ["sample"] = sample,
                ["answered"] = answered,
                ["abstained"] = abstained,
                ["judge_supported_rate_of_answered"] = Rate(supported, answered - unparsable),
                ["judge_unparsable"] = unparsable,
                ["lexical_support_mean"] = lexical.Count == 0 ? null : Math.Round(lexical.Average(), 4),
                ["gold_answer_contained_rate"] = Rate(goldFound, sample),
            };
            Console.WriteLine($"groundedness {name}: {perStore[name]!.ToJsonString()}");
        }

        // Control: each answer is judged against a context from a different question.
        // A useful judge should rarely say YES here.
        var firstStore = storeNames[0];
        int controlYes = 0, controlTotal = 0;
        for (var q = 0; q < sample && sample > 1; q++)
        {
            var top = hits[firstStore][q].Take(ContextSize).ToList();
            var key = q + "|" + string.Join(",", top.Select(h => h.Id));
            if (!answers.TryGetValue(key, out var answer) || Groundedness.IsAbstention(answer)) continue;
            var unrelated = RagService.FormatContext(hits[firstStore][(q + 1) % sample].Take(ContextSize));
            var verdict = Groundedness.ParseVerdict(await Groundedness.JudgeAsync(judge, questions[q].Text, answer, unrelated));
            if (verdict is null) continue;
            controlTotal++;
            if (verdict.Value) controlYes++;
        }

        return new JsonObject
        {
            ["method"] = "Top-5 context per store; answers cached per identical context; a second, larger local model judges YES/NO. Judge and lexical scores are indicative, not ground truth.",
            ["stores"] = perStore,
            ["judge_control_unrelated_context"] = new JsonObject
            {
                ["judged"] = controlTotal,
                ["false_yes_rate"] = Rate(controlYes, controlTotal),
            },
            ["distinct_generations"] = answers.Count,
            ["examples"] = examples,
        };
    }

    private static double? Rate(int a, int b) => b == 0 ? null : Math.Round((double)a / b, 4);
}
