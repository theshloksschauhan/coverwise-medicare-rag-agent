// Example .NET enterprise integration: a Semantic Kernel app that uses CoverWise as a plugin.
//
//   dotnet run -- "Does Medicare cover hearing aids?"
//
// COVERWISE_URL   base URL of the service (default http://localhost:8080)
// OPENAI_API_KEY  optional: if set, an LLM decides when to call the plugin (auto function calling);
//                 otherwise the plugin function is invoked directly through the kernel.
using CoverWise.SkClient;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

var question = args.Length > 0 ? string.Join(' ', args) : "Does Medicare cover hearing aids?";
var baseUrl = Environment.GetEnvironmentVariable("COVERWISE_URL") ?? "http://localhost:8080";
var openAiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");

var builder = Kernel.CreateBuilder();
if (!string.IsNullOrWhiteSpace(openAiKey))
{
    builder.AddOpenAIChatCompletion("gpt-4o-mini", openAiKey);
}
var kernel = builder.Build();
var http = new HttpClient { BaseAddress = new Uri(baseUrl), Timeout = TimeSpan.FromSeconds(60) };
kernel.Plugins.AddFromObject(new CoverWisePlugin(http), "CoverWise");

if (string.IsNullOrWhiteSpace(openAiKey))
{
    var result = await kernel.InvokeAsync("CoverWise", "ask_coverage_question", new() { ["question"] = question });
    var answer = result.GetValue<CoverWiseAnswer>()!;
    Console.WriteLine($"[{answer.Intent}] {answer.Answer}");
    foreach (var c in answer.Citations) Console.WriteLine($"  source: {c.Title} - {c.Source}");
    if (answer.NeedsReview) Console.WriteLine("  (flagged for human review)");
    return answer.Intent is "coverage" or "medical_advice" or "off_topic" ? 0 : 1;
}

var settings = new OpenAIPromptExecutionSettings { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto() };
var reply = await kernel.InvokePromptAsync(
    $"Use the CoverWise plugin to answer, and keep its citations.\nQuestion: {question}", new(settings));
Console.WriteLine(reply);
return 0;
