# Token efficiency of programming languages across tokenizers

Token counts for every solution in [RosettaCodeData](https://github.com/acmeism/RosettaCodeData) (a submodule at `data/RosettaCodeData`), a mirror of [Rosetta Code](https://rosettacode.org) (883 languages, 1,758 tasks, 143,368 files), under the Anthropic, OpenAI and Kimi tokenizers.

[View the interactive comparison](https://zadr.github.io/language-token-comparison/). It is built from the pinned RosettaCodeData commit [`1d475861d7`](https://github.com/acmeism/RosettaCodeData/commit/1d475861d7141ef2fafff293d9339f56e4b0f5db) and published from the [`gh-pages`](https://github.com/zadr/language-token-comparison/tree/gh-pages) branch.

## Tokenizers

| Vendor | Tokenizer | Models | Counted with |
|---|---|---|---|
| Anthropic | Claude | Opus 5.5, Opus 5, Sonnet 5, Fable 5.1 | [`count_tokens`](https://platform.claude.com/docs/en/build-with-claude/token-counting) API, model `claude-opus-5` |
| OpenAI | o200k_base | GPT-5.x | [tiktoken](https://github.com/openai/tiktoken) |
| Moonshot | Kimi | K2 through K3, which share one [`tiktoken.model`](https://huggingface.co/moonshotai/Kimi-K3/blob/main/tiktoken.model) | [tiktoken](https://github.com/openai/tiktoken) |

## Method

- The unit is a task/language pair. Where a pair has several variant solutions, the page uses variant 1 by default, or the mean of all variants as selected.
- Common-task averages take the mean over tasks solved by every selected language.
- The ranking of all languages fits two-way fixed effects on log tokens (task + language) by alternating means. "Typical task" is exp(language effect + mean task effect); the standard error is the language's residual sd / √n.
- `count_tokens` counts a whole message. Its framing overhead (6 tokens for `claude-opus-5`) is measured as count("x") − 1 and subtracted from every file.

## Build

Requires `uv`, `node`, `git` and `curl`.

```sh
make local      # fetch dataset submodule, create venv, inventory files, count o200k and Kimi
make anthropic  # count Claude; needs ANTHROPIC_API_KEY
make page       # write dist/index.html
make test       # check web/model.js against an independent numpy fit
```

`make anthropic` sends one request per unique file (142,656), paced below the organization's `count_tokens` rate limit (`--rpm`, default 7,800), and resumes where it stopped.

## Layout

```
scripts/inventory.py           solution files -> build/files.jsonl
scripts/tokenize_local.py      o200k and Kimi counts
scripts/tokenize_anthropic.py  Claude counts
scripts/build_data.py          columnar data, embedded into dist/index.html
scripts/check_fe.py            numpy fit used by the tests
scripts/shot.sh                headless Chrome screenshots
web/index.html                 page template
web/model.js                   statistics
```
