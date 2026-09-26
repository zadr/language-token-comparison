"""Walk RosettaCodeData/Task and write one row per solution file to build/files.jsonl."""
import hashlib
import json
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "RosettaCodeData"
OUT = ROOT / "build"


def lang_display_names() -> dict[str, str]:
    names = {}
    for meta in (DATA / "Lang").glob("*/00-META.yaml"):
        m = re.search(r"Category:(\S+)", meta.read_text(errors="replace"))
        if m:
            names[meta.parent.name] = urllib.parse.unquote(m.group(1)).replace("_", " ")
    return names


def variant_index(task: str, stem: str) -> int | None:
    """Files are named <task>[-N].<ext>; None marks the unsuffixed form."""
    rest = stem.lower().removeprefix(task.lower())
    return int(rest[1:]) if rest.startswith("-") and rest[1:].isdigit() else None


def main() -> None:
    OUT.mkdir(exist_ok=True)
    names = lang_display_names()
    rows = 0
    with open(OUT / "files.jsonl", "w") as out:
        for task_dir in sorted((DATA / "Task").iterdir()):
            if not task_dir.is_dir():
                continue
            for lang_dir in sorted(task_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                sols = [f for f in sorted(lang_dir.iterdir()) if f.is_file() and not f.name.startswith("00-")]
                # Some pairs keep a stale unsuffixed file beside numbered variants; the numbered set is current.
                if any(variant_index(task_dir.name, f.stem) for f in sols):
                    sols = [f for f in sols if variant_index(task_dir.name, f.stem)]
                for f in sols:
                    raw = f.read_bytes()
                    text = raw.decode("utf-8", errors="replace")
                    if not text.strip():
                        continue
                    sha = hashlib.sha256(text.encode()).hexdigest()
                    out.write(json.dumps({
                        "task": task_dir.name,
                        "lang": names.get(lang_dir.name, lang_dir.name),
                        "lang_dir": lang_dir.name,
                        "variant": variant_index(task_dir.name, f.stem) or 1,
                        "path": str(f.relative_to(DATA)),
                        "sha": sha,
                        "bytes": len(raw),
                        "chars": len(text),
                        "lines": text.count("\n") + 1,
                    }) + "\n")
                    rows += 1
    print(f"{rows} files", file=sys.stderr)


if __name__ == "__main__":
    main()
