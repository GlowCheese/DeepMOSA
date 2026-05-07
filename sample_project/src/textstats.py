import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def word_frequencies(text: str) -> dict[str, int]:
    tokens = tokenize(text)
    return dict(Counter(tokens))


def top_n(text: str, n: int) -> list[tuple[str, int]]:
    if n <= 0:
        raise ValueError("n must be positive")
    freq = word_frequencies(text)
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def text_complexity_grade(text: str) -> str:
    tokens = tokenize(text)
    n = len(tokens)
    if n == 0:
        return "empty"
    if n < 5:
        return "trivial"
    unique_ratio = len(set(tokens)) / n
    avg_len = sum(len(t) for t in tokens) / n
    if unique_ratio > 0.85 and avg_len > 6:
        return "dense"
    if unique_ratio > 0.6 and n > 20:
        return "varied"
    if avg_len > 5:
        return "wordy"
    if unique_ratio < 0.3:
        return "repetitive"
    return "plain"


def extract_hashtags(text: str) -> list[str]:
    if not text:
        return []
    candidates = re.findall(r"#([A-Za-z0-9_]+)", text)
    seen: set[str] = set()
    result: list[str] = []
    for raw in candidates:
        if raw.isdigit():
            continue
        if len(raw) < 2 or len(raw) > 30:
            continue
        if raw.startswith("_") or raw.endswith("_"):
            continue
        tag = raw.lower()
        if tag in seen:
            continue
        seen.add(tag)
        result.append(tag)
    return result
