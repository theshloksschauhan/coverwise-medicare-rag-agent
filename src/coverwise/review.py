"""Human-in-the-loop review queue for low-confidence or uncited answers."""
import threading
from collections import deque
from datetime import UTC, datetime

from coverwise.schemas import ReviewItem

_queue: deque = deque(maxlen=500)
_lock = threading.Lock()


def escalate_to_human(request_id: str, question_masked: str, answer: str, reason: str) -> ReviewItem:
    item = ReviewItem(
        request_id=request_id, question_masked=question_masked, answer=answer, reason=reason,
        created_at=datetime.now(UTC).isoformat(),
    )
    with _lock:
        _queue.append(item)
    return item


def pending() -> list[ReviewItem]:
    with _lock:
        return list(_queue)


def clear() -> None:
    with _lock:
        _queue.clear()
