from __future__ import annotations

import re

from structmem.types import Role


ROLE_PATTERNS: tuple[tuple[Role, re.Pattern[str]], ...] = (
    (Role.TERMINATION, re.compile(r"\b(done|finish|final answer|task complete|submit)\b", re.I)),
    (Role.RECOVERY, re.compile(r"\b(retry|back|undo|recover|instead|failed|can't|cannot)\b", re.I)),
    (Role.VERIFICATION, re.compile(r"\b(check|inspect|measure|examine|verify|observe|inventory)\b", re.I)),
    (Role.PRECONDITION_CHECK, re.compile(r"\b(open|find|locate|ensure|whether|available|relations)\b", re.I)),
    (Role.INFORMATION_ACQUISITION, re.compile(r"\b(look|search|query|get|read|list|go to|navigate)\b", re.I)),
)


def annotate_action(action: str, observation: str = "") -> Role:
    text = f"{action} {observation}"
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(text):
            return role
    return Role.STATE_CHANGING_ACTION


def annotate_trace(actions: list[str], observations: list[str]) -> list[Role]:
    return [
        annotate_action(action, observations[index] if index < len(observations) else "")
        for index, action in enumerate(actions)
    ]

