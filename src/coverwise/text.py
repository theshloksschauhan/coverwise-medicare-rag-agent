import re

_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "by", "with", "my", "i",
    "do", "does", "can", "if", "it", "this", "that", "you", "your", "me", "what", "how", "will", "at", "as",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]
