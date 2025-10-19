import json


def append_jsonl(path: str, obj) -> None:
    """Appends to jsonl. (native to io_jsonl.py). Does not delete."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
