namespace KernelRag.Core;

public static class Metrics
{
    /// <summary>1 if the gold id is in the first k results, else 0.</summary>
    public static int HitAt(IReadOnlyList<string> ranked, string gold, int k) =>
        ranked.Take(k).Contains(gold) ? 1 : 0;

    public static double ReciprocalRank(IReadOnlyList<string> ranked, string gold)
    {
        for (var i = 0; i < ranked.Count; i++)
            if (ranked[i] == gold) return 1.0 / (i + 1);
        return 0;
    }

    /// <summary>Nearest-rank percentile.</summary>
    public static double Percentile(IEnumerable<double> values, double p)
    {
        var sorted = values.OrderBy(v => v).ToArray();
        if (sorted.Length == 0) throw new ArgumentException("no values", nameof(values));
        var rank = (int)Math.Ceiling(p / 100 * sorted.Length);
        return sorted[Math.Clamp(rank - 1, 0, sorted.Length - 1)];
    }

    /// <summary>95% Wilson score interval for a proportion.</summary>
    public static (double Low, double High) Wilson(int successes, int n)
    {
        if (n == 0) return (0, 0);
        const double z = 1.96;
        double p = (double)successes / n, z2 = z * z;
        var center = (p + z2 / (2 * n)) / (1 + z2 / n);
        var margin = z * Math.Sqrt(p * (1 - p) / n + z2 / (4.0 * n * n)) / (1 + z2 / n);
        return (Math.Max(0, center - margin), Math.Min(1, center + margin));
    }
}
