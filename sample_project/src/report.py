from datetime import datetime, timezone

from dateutil import parser as dateparser
from tabulate import tabulate

from .textstats import top_n, word_frequencies


def render_top_table(text: str, n: int) -> str:
    rows = top_n(text, n)
    return tabulate(rows, headers=["word", "count"], tablefmt="github")


def total_unique_words(text: str) -> int:
    return len(word_frequencies(text))


def summary_line(text: str) -> str:
    unique = total_unique_words(text)
    if unique == 0:
        return "empty document"
    return f"{unique} unique words"


def normalize_event_timestamp(raw: str) -> str:
    if not raw or not raw.strip():
        raise ValueError("empty timestamp")
    try:
        dt = dateparser.isoparse(raw)
    except (ValueError, TypeError):
        try:
            dt = dateparser.parse(raw, fuzzy=False)
        except (ValueError, TypeError, OverflowError):
            try:
                dt = datetime.strptime(raw.strip(), "%d/%m/%Y %H:%M")
            except ValueError as e:
                raise ValueError(f"unrecognized timestamp: {raw!r}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def bucket_score(score: float) -> str:
    if score != score:
        raise ValueError("score is NaN")
    if score < 0 or score > 100:
        raise ValueError(f"score out of range: {score}")
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def render_event_summary(events: list[tuple[str, float]]) -> str:
    if not events:
        return "no events"
    rows: list[tuple[str, float, str]] = []
    skipped = 0
    for ts, score in events:
        try:
            normalized = normalize_event_timestamp(ts)
        except ValueError:
            skipped += 1
            continue
        grade = bucket_score(score)
        rows.append((normalized, score, grade))
    if not rows:
        return f"no valid events ({skipped} skipped)"
    rows.sort(key=lambda r: r[0])
    table = tabulate(rows, headers=["time", "score", "grade"], tablefmt="github")
    if skipped:
        table += f"\n({skipped} skipped)"
    return table
