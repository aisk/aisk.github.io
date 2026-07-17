# The Mention File Myth in Code Agents

> *Disclaimer: This post was written by me, but I used GLM 5.6 to restructure and polish the text. Parts of the investigation behind it were also done with GLM 5.6.*

## How it started

When Claude Code first came out, I picked up two habits from the official blog post[^1] that immediately felt like upgrades: using `!` to run shell commands inline, and using `@` to mention a file.

Both felt handy. `!` let me run commands I was comfortable with myself and feed the output into Claude's context, or simply take over the follow-up work by hand while Claude kept the full context of what happened. `@` felt just as good: it let me tell Claude exactly where a file was. If I only gave it a file name, it would have to generate content to call tools and go look for the file, the location might not be accurate, and it could take several rounds of operations before it found the right one.

For a while, this was simply how I worked.

## The "wasted tokens" take

A few months later, I started seeing people on X arguing against `@`. The argument was simple and, at the time, felt obviously correct: `@` is not magic. It is a deterministic piece of client-side code that reads the file from disk, expands its contents into the user's prompt, and sends the whole thing to the LLM.[^2] So if you `@` a very large file, you are wasting an enormous amount of tokens.

That clicked for me. Of course that is how mention file had to be implemented: like `!` invoking the shell, it is at heart a deterministic program, so this kind of problem is hard to avoid.

## The vibe coding shift

Then vibe coding became a thing, models got noticeably stronger, and I found myself no longer using `@` most of the time. Every time I watched the model work, it checked the directory structure whenever it needed to, and when I asked it to read some file, it usually already knew where the file was. Context windows had also grown across the board, so letting the model spend a few extra tool calls and round trips was not much of a cost; in exchange, I got the efficiency of describing files vaguely instead of typing exact paths. (Yes: I had discovered that the bottleneck was now me, and my typing speed.)

I didn't think about how `@` was implemented again for a long time.

## Building my own agent

That changed when I started building [paimon](https://github.com/aisk/paimon), my own code agent. When it came time to implement mention file, the old question came back to me. I thought it through carefully, and suddenly it seemed the old conclusion wasn't necessarily right.

- What if `@` is just a local convenience for the user at input time (an autocomplete for file paths), and the program never reads any content into the prompt at all?
- Or, thinking about it more: mention file really is a deterministic behavior. If the code agent has actually read a file once, could it send the full content along with more information, like a sha256 of the contents? Then the next time the file is read, if it finds it was sent before and the sha256 hasn't changed, could it send only the metadata, letting the model know it has already read this file and the hash is unchanged?
- And if so, when the content does need to be expanded, how should compaction be handled? Would this need some complicated machinery to manage?

After thinking it over, I came up with a design based on passing metadata. Each mentioned file would be wrapped in an XML envelope that tells the model the file path, a sha256 of the contents, whether the enclosed content is the full file or only part of it, which lines are included, and how many lines the file actually has. Roughly:

```xml
<file path="src/parser.py" sha256="3f6a…" content="partial" lines="1-200" total-lines="1240">
...
</file>
```

If the full content had already been sent in the current session and the hash was unchanged, only the reference metadata would go out, letting the model know it had already read this file and nothing had changed; files over a size threshold would only get their head sent, with the real length in the metadata. Since no model has ever seen this convention, the rules would be spelled out in the system prompt. Underneath, the agent would keep an in-memory table of every file sent so far (the hash, and what exactly had been transmitted), and reset it after compaction, when "already sent in full" no longer holds. It added up to a rather complex state transition diagram.

It sounds like a bit of a hassle, but with the help of ChatGPT 5.6 sol, the states and flows got sorted out quickly. It even thoughtfully added one more design on top: when the user pins line numbers with the `@filename:10-20` syntax, the fully transmitted ranges are recorded too; when a later mention of another range overlaps with what was sent before and is fully covered, the ranges can be merged. Computing those overlaps (which ranges are already covered, which partially intersect, which should be merged into one) was the genuinely fiddly part of the whole design.

## Wait, didn't Cursor try this?

That was the moment it suddenly hit me: isn't this mechanism getting way too complex?

It reminded me of when Cursor first launched, and people were analyzing and guessing at how it must work under the hood. The story went: LLM context windows were small, so the code was retrieved with RAG and only the relevant snippets were sent[^3]; some analyses went further, speculating that after a tool call finished operating on files or commands, the context around the call was dropped entirely, with just the result stitched back in. Whether or not that second part was ever true, the design people believed in looked extremely clever. But in my own use at the time, the results on longer agent-style tasks were underwhelming. Then Claude Code came out and did almost the opposite: it hunted for code bluntly with `rg` and other plain commands, never rolled context back, and just kept pushing everything forward until the window filled up and compaction kicked in. For me it worked dramatically better.[^4]

I don't claim to fully understand how LLMs work, but seeing Claude Code's approach made me realize what that rumored design was essentially doing: treating the context window as a kind of database, assuming that anything sent to the LLM would be reliably remembered. But the attention-based LLMs popular today make no such guarantee. Content that has already appeared in the context is not reliably used, especially once it sits deep in a long history[^5], which is why the model will try to read things again.

And now I was designing exactly that kind of clever mechanism. The overlap merging, the hash-based dedup, the "just send metadata if the hash matches" trick. All of it was treating the context window as a database of record, and maybe I was making the same mistake.

## Checking the facts

Fortunately, open-source code agents are everywhere now, and quite a few of them are well known. So instead of theorizing, I decided to actually investigate whether the conclusions I had reached last year were true. Using opencode with GLM 5.6, I went through five well-known open-source agents: `pi`, `opencode`, `gemini-cli`, `grok-build`, and `codex`.[^6]

Even the very first question, whether `@` sends file contents at all, splits the field. Four of the five read the file and inline it into the user message: pi wraps it in `<file name="...">` (though only for files passed as CLI startup arguments; in its interactive TUI, `@` is just path autocomplete and the literal path text is sent), grok-build in `<file_contents path="...">` with line numbers, gemini-cli between `--- Content from referenced files ---` markers, and opencode fakes a tool call — the model sees `Called the Read tool with the following input: {...}` followed by standard Read output, as if it had called the tool itself. And then there is codex, which does not read the file at all: `@` is a fuzzy filename search, and picking a result just inserts the path as plain text. The model is expected to `cat` or `rg` it when it cares. That is exactly the "maybe `@` is just autocomplete" possibility from my first bullet point.

| Agent | What `@` sends | Metadata beyond the path | Line-range syntax | Tracks what was sent |
|---|---|---|---|---|
| **pi** | Full content, no truncation (CLI args only) | None | ❌ | ❌ |
| **opencode** | Content through its own Read tool (2000 lines / 50 KB cap), faked as a tool call | Truncation notes: "Use offset=N to continue" | ✅ `@file#12-18` | ❌ |
| **gemini-cli** | Full content (2000-line cap, 20 MB hard reject) | A truncation warning pointing at `read_file` | ❌ | ❌ |
| **grok-build** | Full content up to ~5,000 est. tokens, then a metadata-only stub | `skipped="true"` + reason on oversized files | ✅ `@foo.rs:10-20` | ❌ |
| **codex** | Nothing, just the path as plain text | — | ❌ | ❌ |

Then I held my paimon design up against these implementations, point by point.

**Nobody sends a hash, and a file's true size surfaces only in truncation hints.** The envelopes carry a path and the content, and that is essentially it (opencode's Read-style output does close with a total line count). The closest thing to my metadata design is grok-build's handling of oversized files: past ~5,000 estimated tokens it drops the body entirely and sends a stub like `<file_contents path="..." skipped="true" reason="file too large (~5800 estimated tokens, limit 5000). Use read_file tool to read specific sections."/>`. The truncating agents all do some version of this: tell the model it got cut off and point it back at its own read tool. Metadata as a "go read it yourself" hint, never as a dedup key.

**Nobody tracks what was already sent.** No hash, no mtime, no in-memory table, no "you already have this file" path. Every mention re-reads the disk and re-sends in full; the only dedup anywhere is a `Set` that collapses duplicate mentions within a single message. The one sha256 I found in a mention path (grok-build's) is used to name spill files on disk and never reaches the model. The one genuine counterexample is buried in gemini-cli: a `ContextCompressionService` that hashes file contents and asks a small model to route each old file to FULL / PARTIAL / SUMMARY / EXCLUDED, startlingly close to a chunk of my design. Except it sits behind a default-off experimental flag, nothing in the runtime actually instantiates it at the commit I read, and even if it ran, it only processes read-tool responses, not `@`-mention inlines. Someone had the same idea, and it has not shipped.

**Line-range syntax exists; overlap bookkeeping does not.** opencode supports `@file#12-18` (translated into a Read call with offset and limit) and grok-build supports `@foo.rs:10-20`. Neither records which ranges were sent, and neither merges overlapping ones; every mention is an independent read. The overlap-merging machinery ChatGPT and I sketched exists nowhere.

**Nobody explains the convention in the system prompt.** All five system prompts are silent about what a mention looks like. pi and grok-build rely on the XML being self-describing; a comment in grok-build's source calls its format "the training format we have been using". opencode's fake Read call is the cleverest dodge: nothing needs to be documented, because the model already knows what Read output looks like. And codex has nothing to explain, because it sends nothing but a path.

**Compaction does nothing special with mentions.** In all five, when history gets summarized, inlined file content is treated as ordinary user text and fed to the summarizer wholesale: no hash routing, no placeholder substitution. My worry about "how does the state survive compaction" dissolves completely: there is no state to survive.

**Nobody watches for staleness.** If a mentioned file changes on disk afterwards, nothing marks the old copy in the context as expired. The model finds out whenever it next happens to read the file.

**And the most telling discovery: codex used to do it the other way.** In its TypeScript CLI era, codex's `@` worked exactly like the others: "file contents automatically expanded into XML blocks before being sent to the LLM", in the words of the pull request that introduced it, with `@path[50:80]` line selection listed as a next step. The Rust rewrite replaced all of that with the path-only fuzzy search, without a word in the commit history about why content expansion was dropped, and the TypeScript implementation was later deleted outright. The only agent that demonstrably walked the "expand and enrich" road turned around and walked all the way back to the most minimal design possible.

## Closing thought

So my old understanding was wrong on just about every count. In one direction, today's code agents are smarter than I imagined: most of them will not naively shovel an entire file at the LLM in one go; past a certain size they truncate, refuse, or swap in a stub that tells the model to go read the file itself. In the other direction, none of them needed the elaborate system I had been so pleased with designing. However precise the state diagram looked, the LLM does not want any of it. The model does not fail because we re-sent a file; it fails when it cannot find the file, or when we sent less than it needed. The complexity was for my own satisfaction, not the model's.

As for where this is heading: models keep getting smarter, and codex's design is a straight bet on that. Send nothing but the filename, and let the model fetch whatever it cares about. Given how well agentic search already works, I suspect this is the better way, and I would not be surprised to see more agents drift toward it.

And there is a bigger difficulty hiding behind this small one, which I only appreciated by building an agent myself. Questions like "expand the file or just send the path" cannot be settled by reading code or by taste: you have to evaluate them against real tasks, and evaluation burns tokens at a scale my previous side projects never did. A CLI tool can be verified locally in seconds, for free; an agent design decision costs real money per data point.

Worse, the answer may not transfer between models. Reinforcement learning is doing a lot of the work in making models better, and it also stamps each model with its own working style, so each vendor's CLI will naturally ship whatever mechanism its own model was trained toward and scores best on its own evals. grok-build's source says this out loud: its XML format exists because it is "the training format we have been using". For anyone building a general-purpose agent on top of other people's models, that is a quiet, permanent source of trouble: the best mechanism is not universal, and you cannot afford to measure everything.

[^1]: "Claude Code: Best practices for agentic coding", originally published on Anthropic's engineering blog, now maintained as part of the [official docs](https://code.claude.com/docs/en/best-practices).

[^2]: The official docs still describe `@` this way: "Reference files with `@` instead of describing where code lives. Claude reads the file before responding."

[^3]: This part of the guess was roughly right, except for the "local": Cursor chunks files locally, but computes embeddings on its servers and stores them in a remote vector database, while the code itself stays on your machine. See [How Cursor Indexes Codebases Fast](https://read.engineerscodex.com/p/how-cursor-indexes-codebases-fast). The "drop the tool-call context" part, as far as I can tell, was never confirmed; early Cursor didn't even have agent-style tool calls.

[^4]: A detail I only learned while fact-checking this post: early Claude Code had also tried RAG with a local vector database, and [dropped it in favor of plain agentic search](https://vadim.blog/claude-code-no-indexing/). Boris Cherny, its creator: "Early versions of Claude Code used RAG + a local vector db, but we found pretty quickly that agentic search generally works better." Another Anthropic engineer in the same thread: "In our testing we found that agentic search outperformed [it] by a lot, and this was surprising."

[^5]: This is a measured phenomenon, not just a vibe: models use information in the middle of a long context significantly worse than information at the beginning or end. See [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172).

[^6]: All five checked in mid-July 2026, at commits: pi `87ad8243`, opencode `efb6cc2d4`, gemini-cli `3ff5ba2`, grok-build `98c3b24`, codex `315195492c`. These are moving targets; details below may have changed by the time you read this.
