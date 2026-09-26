"""Count tokens for every unique solution with the locally available tokenizers.

Writes build/counts/<tokenizer>.jsonl as {"sha", "n"} rows.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "RosettaCodeData"
COUNTS = ROOT / "build" / "counts"

_enc = {}


def _load():
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    kimi_pat = "|".join([
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ])
    kimi = tiktoken.Encoding(
        name="kimi",
        pat_str=kimi_pat,
        mergeable_ranks=load_tiktoken_bpe(str(ROOT / "tok" / "kimi" / "tiktoken.model")),
        special_tokens={},
    )
    o200k = tiktoken.get_encoding("o200k_base")
    _enc.update({
        "openai-o200k": lambda t: len(o200k.encode_ordinary(t)),
        "kimi": lambda t: len(kimi.encode_ordinary(t)),
    })


def count(item):
    sha, path = item
    if not _enc:
        _load()
    text = (DATA / path).read_bytes().decode("utf-8", errors="replace")
    return sha, {k: f(text) for k, f in _enc.items()}


def main() -> None:
    COUNTS.mkdir(parents=True, exist_ok=True)
    unique = {}
    for line in open(ROOT / "build" / "files.jsonl"):
        r = json.loads(line)
        unique.setdefault(r["sha"], r["path"])
    outs = {}
    with Pool() as pool:
        for i, (sha, counts) in enumerate(pool.imap_unordered(count, unique.items(), chunksize=256)):
            for k, n in counts.items():
                if k not in outs:
                    outs[k] = open(COUNTS / f"{k}.jsonl", "w")
                outs[k].write(json.dumps({"sha": sha, "n": n}) + "\n")
            if i % 20000 == 0:
                print(i, file=sys.stderr)
    for f in outs.values():
        f.close()


if __name__ == "__main__":
    main()
