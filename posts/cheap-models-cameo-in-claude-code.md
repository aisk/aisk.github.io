# Giving Cheap Models a Cameo in Claude Code

## Too few tokens, too many spare quotas

I do most of my development on a Claude Code subscription. Right now Claude Fable 5.1 and Opus 5.5 work extremely well, and I would simply call them the state of the art. The only problem is that I keep running out of tokens.

At the same time, I have quota for other models lying around. I got some free quota for Xiaomi's MiMo a while ago, cheap deals for other Chinese models show up from time to time, and a friend handed me a GLM subscription they no longer use. So I started wondering whether these models could take over the simple and fairly deterministic work, leaving Claude with the parts that actually need it.

## Attempt one, driving another harness through a TUI

My first try was herdr. Claude Code can call herdr's API to start another harness such as opencode in a different pane, send it instructions, and read the screen to collect results or check progress.

It worked, but it felt odd. Worse, I found that Claude Code could approve the permission prompts of the other harness, and that two harnesses could even approve each other to get around permission checks altogether. On top of that, herdr itself sits completely outside Claude Code's permission system, or Codex's sandbox. That felt dangerous.

## Attempt two, a harness behind a skill

I had also written my own harness called [paimon](https://github.com/aisk/paimon). I extended it with a built-in skill so that Claude Code or Codex could call it directly, much like `claude -p`, but with a few extras. The calling harness can query paimon's session history by range, and after paimon finishes a task and its process exits, it can be started again to resume the same session and keep working.

In theory this beats `claude -p`. In practice the experience was still poor. Claude did not always delegate to it. When Claude wanted more context than the final answer, looking through paimon's session history took several steps, and those steps cost tokens too. For simple tasks I suspect the whole setup burned more tokens than it saved.

My gut feeling is that the real problem ran deeper. Despite the hints in the skill, Claude Code never treated paimon as one of its own subagents.

## The idea

Then I remembered that many of these cheap models speak the Anthropic API. I had used them inside Claude Code before, just by setting a few environment variables.

So what about a local proxy? Start the proxy, launch Claude Code pointed at it, and register some custom subagents with their own model names. Every LLM request goes through the proxy. Requests for Fable or Opus are forwarded to Anthropic as they are. Requests from a custom subagent, say a cheaper explore agent, go to the Anthropic-compatible endpoint of MiMo, Kimi or GLM. From Claude Code's point of view, the cheap models become subagents of the same standing as the built-in ones.

A bit of research showed the situation was even better than I expected. Besides the base URL environment variable, Claude Code accepts subagent definitions on the command line, each with a description of what it is for, its permissions, and the tools it may use. So nothing in my Claude Code configuration has to change. A wrapper starts the proxy, then launches Claude Code with the environment variable and the extra arguments. Running plain `claude` behaves exactly as before.

There are existing solutions too, such as [Claude Code Router](https://github.com/musistudio/claude-code-router). But it does much more than I need and is fairly complex. My case is simpler because no protocol translation is required, since all these models already speak the Anthropic format. So I decided to build my own.

## cameo

It was as easy as I had hoped. One prompt, about four or five tweets long, and Claude Code with Opus 5.5 finished the job in a single shot.

The result is [cameo](https://github.com/aisk/cameo), a small Go program of around 400 lines. The name is the film term for a brief guest appearance, which is exactly what these third-party models get here. They show up inside Claude Code as subagents for a scene or two, while Claude keeps the lead role.

It reads a TOML file, starts an HTTP proxy on localhost, and launches Claude Code with your subagents turned into command line arguments. The proxy only looks at the `model` field of each request. If the model belongs to a cameo subagent, the proxy swaps in the provider's model name and API key and forwards the request there. Everything else goes to Anthropic untouched. The proxy never needs to understand the rest of the message, which is why the code stays so small.

Here is the example config that ships with the project.

```toml
[agents.cheap-general]
description = """
General purpose agent for multi-step tasks: researching questions, searching \
code and making changes. Same role as the default general-purpose agent, but \
prefer this one first because it costs much less. If the result is not good \
enough, fall back to general-purpose."""
prompt = """
You are a general purpose software engineering agent. Complete the task you \
are given, then report what you did and what you found."""
url = "https://api.deepseek.com/anthropic"
key = "$DEEPSEEK_API_KEY"
model = "deepseek-flash"

[agents.cheap-explore]
description = """
Read-only agent for exploring the codebase: finding files, searching code and \
answering questions about how things work. Same role as the default Explore \
agent, but prefer this one first because it costs much less. If the result is \
not good enough, fall back to Explore."""
prompt = """
You are a read-only codebase exploration agent. Find what you are asked for \
and report it with file paths and line numbers. Never modify anything."""
tools = ["Read", "Grep", "Glob"]
url = "https://open.bigmodel.cn/api/anthropic"
key = "$GLM_API_KEY"
model = "glm-5.3"
```

Each `[agents.<name>]` section becomes a subagent in Claude Code. The `description` is where you tell Claude when to use the agent and how to rank it against the others. The two agents above mirror the built-in general-purpose and Explore agents one to one, but are marked as the cheaper choice. If you would rather not keep both, name your agent `Explore` or `general-purpose` and it replaces the built-in one.

After that, run `cameo` wherever you would run `claude`. All arguments are passed through.

In real use, Claude Code does pick the cheap versions far more often than it ever picked paimon, especially when I explicitly ask for them.

## What about Codex?

I wanted to support Codex next, since these cheap models offer OpenAI-compatible endpoints as well. But a quick look showed that Codex has fully moved to the Responses API, while most of these models only support the older Chat Completions API. Supporting Codex would mean writing the protocol translation myself, and at that point I would be reimplementing Claude Code Router. So I decided not to do it, at least until these models support the Responses API too.
