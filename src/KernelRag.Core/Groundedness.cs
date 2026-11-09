using System.Text.RegularExpressions;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Ollama;

namespace KernelRag.Core;

public static partial class Groundedness
{
    private static readonly HashSet<string> StopWords = new(
        "a an and are as at be by for from has have in is it its of on or that the this to was were which with who what when where why how".Split(' '));

    [GeneratedRegex(@"[\p{L}\p{N}]+")]
    private static partial Regex Word();

    public static IReadOnlyList<string> Tokens(string text) =>
        Word().Matches(text.ToLowerInvariant()).Select(m => m.Value).ToList();

    public static bool IsAbstention(string answer) =>
        answer.Contains("don't know", StringComparison.OrdinalIgnoreCase) ||
        answer.Contains("don’t know", StringComparison.OrdinalIgnoreCase) ||
        answer.Contains("do not know", StringComparison.OrdinalIgnoreCase);

    /// <summary>Share of the answer's content words that also occur in the context (no LLM involved).</summary>
    public static double LexicalSupport(string answer, string context)
    {
        var contextWords = Tokens(context).ToHashSet();
        var content = Tokens(answer).Where(t => !StopWords.Contains(t)).ToList();
        return content.Count == 0 ? 0 : content.Count(contextWords.Contains) / (double)content.Count;
    }

    /// <summary>True if a normalised gold answer occurs as a word sequence in the answer.</summary>
    public static bool ContainsGold(string answer, IEnumerable<string> golds)
    {
        var haystack = " " + string.Join(' ', Tokens(answer)) + " ";
        return golds.Any(g => haystack.Contains(" " + string.Join(' ', Tokens(g)) + " ", StringComparison.Ordinal));
    }

    /// <summary>Parses the judge reply. Null means the judge gave no usable verdict.</summary>
    public static bool? ParseVerdict(string reply)
    {
        var first = Tokens(reply).FirstOrDefault();
        return first switch { "yes" => true, "no" => false, _ => null };
    }

    private const string JudgePrompt = """
        <message role="system">You check whether an answer is fully supported by a context. Reply with one word: YES if every claim in the answer is stated in the context, otherwise NO.</message>
        <message role="user">Context:
        {{$context}}

        Question: {{$question}}
        Answer: {{$answer}}

        Is the answer fully supported by the context? Reply YES or NO.</message>
        """;

    public static async Task<string> JudgeAsync(Kernel kernel, string question, string answer, string context,
        CancellationToken ct = default)
    {
        var settings = new OllamaPromptExecutionSettings { Temperature = 0, NumPredict = 8 };
        var function = kernel.CreateFunctionFromPrompt(JudgePrompt, settings);
        var result = await kernel.InvokeAsync(function, new KernelArguments(settings)
        {
            ["context"] = context, ["question"] = question, ["answer"] = answer,
        }, ct);
        return result.ToString();
    }
}
