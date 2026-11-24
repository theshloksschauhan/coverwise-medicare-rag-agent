using System.ComponentModel;
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using Microsoft.SemanticKernel;

namespace CoverWise.SkClient;

/// <summary>Semantic Kernel plugin that lets any .NET app call the CoverWise Python service.</summary>
public sealed class CoverWisePlugin(HttpClient http)
{
    [KernelFunction("ask_coverage_question")]
    [Description("Answers a Medicare coverage question with citations. Not medical advice.")]
    public async Task<CoverWiseAnswer> AskAsync(
        [Description("The coverage question, e.g. 'Does Medicare cover hearing aids?'")] string question,
        CancellationToken cancellationToken = default)
    {
        using var response = await http.PostAsJsonAsync("/ask", new { question }, cancellationToken);
        response.EnsureSuccessStatusCode();
        return (await response.Content.ReadFromJsonAsync<CoverWiseAnswer>(cancellationToken))!;
    }
}

public sealed record Citation(
    [property: JsonPropertyName("chunk_id")] string ChunkId,
    [property: JsonPropertyName("title")] string Title,
    [property: JsonPropertyName("source")] string Source);

public sealed record CoverWiseAnswer(
    [property: JsonPropertyName("request_id")] string RequestId,
    [property: JsonPropertyName("intent")] string Intent,
    [property: JsonPropertyName("answer")] string Answer,
    [property: JsonPropertyName("citations")] List<Citation> Citations,
    [property: JsonPropertyName("confidence")] double Confidence,
    [property: JsonPropertyName("needs_review")] bool NeedsReview,
    [property: JsonPropertyName("refusal_reason")] string? RefusalReason);
