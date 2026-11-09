namespace KernelRag.Core;

public static class Chunker
{
    /// <summary>Splits text into overlapping windows of whitespace-separated words.</summary>
    public static IReadOnlyList<string> Chunk(string text, int maxWords = 200, int overlapWords = 30)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxWords);
        if (overlapWords < 0 || overlapWords >= maxWords)
            throw new ArgumentOutOfRangeException(nameof(overlapWords));

        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var chunks = new List<string>();
        for (var start = 0; start < words.Length; start += maxWords - overlapWords)
        {
            chunks.Add(string.Join(' ', words.Skip(start).Take(maxWords)));
            if (start + maxWords >= words.Length) break;
        }
        return chunks;
    }
}
