from __future__ import annotations

import json

from structmem.types import Experience


SYSTEM = """You are an agent acting in an interactive benchmark. Follow the environment's exact action syntax. Return only one executable action, with no explanation or prefix. Never claim success; the environment is the only success judge."""


def compact_experience(experience: Experience) -> str:
    return json.dumps(
        {
            "id": experience.experience_id,
            "source": experience.dataset,
            "goal": experience.goal,
            "roles": [role.value for role in experience.roles],
            "actions": experience.actions,
            "lesson": experience.lesson,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )


def action_messages(
    goal: str,
    observation: str,
    history: list[tuple[str, str]],
    guidance: str = "",
    admissible_actions: list[str] | None = None,
) -> list[dict[str, str]]:
    recent = "\n".join(f"Observation: {obs}\nAction: {action}" for obs, action in history[-6:]) or "None"
    candidates = ""
    if admissible_actions:
        candidates = "\nChoose exactly one of these admissible actions:\n" + "\n".join(admissible_actions)
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Task: {goal}\nCurrent observation: {observation}\nRecent trajectory:\n{recent}"
                f"\nMemory guidance:\n{guidance or 'None'}{candidates}\nNext action:"
            ),
        },
    ]


def extract_action(text: str, admissible_actions: list[str] | None = None) -> str:
    stripped = text.strip()
    if not admissible_actions and (
        stripped.startswith("Action: Operation") or stripped.startswith("Action: Answer")
    ):
        return stripped
    candidate = stripped.splitlines()[0].strip() if stripped else ""
    for prefix in ("Action:", "action:", "Next action:"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :].strip()
    candidate = candidate.strip("`\"'")
    if admissible_actions:
        exact = {action.lower(): action for action in admissible_actions}
        if candidate.lower() in exact:
            return exact[candidate.lower()]
        contained = [action for action in admissible_actions if action.lower() in candidate.lower()]
        if len(contained) == 1:
            return contained[0]
        return "look"
    return candidate or "look"
