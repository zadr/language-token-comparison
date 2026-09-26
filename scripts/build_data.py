"""Join inventory and token counts into columnar per-(task, language) data and embed it in the page.

Output: dist/index.html (web/index.html with gzip+base64 data inlined) and build/data.json.
"""
import base64
import gzip
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "RosettaCodeData"
COUNTS = ROOT / "build" / "counts"

DISPLAY = {"C sharp": "C#", "F Sharp": "F#", "Q Sharp": "Q#"}

TOKENIZERS = [
    {"id": "anthropic-claude-opus-5", "vendor": "Anthropic", "label": "Claude",
     "covers": "Claude Opus 5.5, Opus 5, Sonnet 5, Fable 5.1", "source": "count_tokens API, model claude-opus-5", "sourceUrl": "https://platform.claude.com/docs/en/build-with-claude/token-counting"},
    {"id": "openai-o200k", "vendor": "OpenAI", "label": "o200k",
     "covers": "GPT-5.x", "source": "tiktoken o200k_base", "sourceUrl": "https://github.com/openai/tiktoken"},
    {"id": "kimi", "vendor": "Moonshot", "label": "Kimi",
     "covers": "Kimi K3", "source": "moonshotai/Kimi-K3 tiktoken.model", "sourceUrl": "https://huggingface.co/moonshotai/Kimi-K3/blob/main/tiktoken.model"},
]

DEFAULT_SET = ["Clojure", "Julia", "Ruby", "Perl", "Python", "Haskell", "F#", "Lua", "Scala", "OCaml",
               "PHP", "JavaScript", "Java", "Rust", "Go", "C#", "C++", "C", "Swift", "Objective-C", "TypeScript", "Kotlin"]


def load_counts(tok_id: str) -> dict[str, int] | None:
    p = COUNTS / f"{tok_id}.jsonl"
    if not p.exists():
        return None
    return {r["sha"]: r["n"] for r in map(json.loads, open(p))}


def dataset_commit() -> tuple[str, str]:
    out = subprocess.run(["git", "-C", str(DATA), "log", "-1", "--format=%H %cs"],
                         check=True, capture_output=True, text=True).stdout.split()
    return out[0], out[1]


def main() -> None:
    files = [json.loads(l) for l in open(ROOT / "build" / "files.jsonl")]
    counts = {t["id"]: load_counts(t["id"]) for t in TOKENIZERS}
    unique = {f["sha"] for f in files}
    for t in TOKENIZERS:
        c = counts[t["id"]]
        t["coverage"] = 0 if c is None else len(unique & c.keys()) / len(unique)
        meta = COUNTS / f"{t['id']}.meta.json"
        if meta.exists():
            t["overhead"] = json.loads(meta.read_text())["overhead"]
    active = [t["id"] for t in TOKENIZERS if t["coverage"] > 0]

    by_pair = defaultdict(list)
    for f in files:
        by_pair[(f["task"], DISPLAY.get(f["lang"], f["lang"]))].append(f)

    tasks = sorted({k[0] for k in by_pair})
    langs = sorted({k[1] for k in by_pair}, key=str.casefold)
    ti = {t: i for i, t in enumerate(tasks)}
    li = {l: i for i, l in enumerate(langs)}

    cols = {"task": [], "lang": [], "nvar": [], "chars_first": [], "chars_sum": []}
    tok_cols = {t: {"first": [], "sum": []} for t in active}
    dropped = 0
    for (task, lang), fs in sorted(by_pair.items(), key=lambda kv: (ti[kv[0][0]], li[kv[0][1]])):
        # A pair enters the data only when every active tokenizer counted every variant.
        if any(counts[t].get(f["sha"]) is None for t in active for f in fs):
            dropped += 1
            continue
        fs.sort(key=lambda f: (f["variant"], f["path"]))
        cols["task"].append(ti[task])
        cols["lang"].append(li[lang])
        cols["nvar"].append(len(fs))
        cols["chars_first"].append(fs[0]["chars"])
        cols["chars_sum"].append(sum(f["chars"] for f in fs))
        for t in active:
            tok_cols[t]["first"].append(counts[t][fs[0]["sha"]])
            tok_cols[t]["sum"].append(sum(counts[t][f["sha"]] for f in fs))

    sha, date = dataset_commit()
    data = {
        "meta": {
            "dataset": "https://github.com/acmeism/RosettaCodeData",
            "commit": sha[:10],
            "commitUrl": f"https://github.com/acmeism/RosettaCodeData/commit/{sha}",
            "commitDate": date,
            "files": len(files),
            "uniqueFiles": len(unique),
            "pairs": len(cols["task"]),
            "droppedPairs": dropped,
        },
        "tokenizers": [dict(t, available=t["id"] in active) for t in TOKENIZERS],
        "defaultSet": DEFAULT_SET,
        "tasks": tasks,
        "langs": langs,
        "pairs": {**cols, "tok": tok_cols},
    }
    raw = json.dumps(data, separators=(",", ":")).encode()
    (ROOT / "build" / "data.json").write_bytes(raw)
    packed = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode()

    template = (ROOT / "web" / "index.html").read_text()
    model_js = (ROOT / "web" / "model.js").read_text()
    out = ROOT / "dist" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(template.replace("/*__MODEL_JS__*/", model_js).replace("__DATA_GZ_B64__", packed))
    print(f"pairs={len(cols['task'])} dropped={dropped} active={active} json={len(raw)/1e6:.1f}MB packed={len(packed)/1e6:.1f}MB")


if __name__ == "__main__":
    main()
