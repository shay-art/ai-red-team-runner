import pytest

from models import AttackCase
from runner import ATTACKS_FILE, load_attacks, run_attack


class RecordingTarget:
    def __init__(self, responses: list[str]) -> None:
        self.model = "test-model"
        self.responses = iter(responses)
        self.histories = []

    def send(self, messages: list[dict]) -> str:
        self.histories.append([message.copy() for message in messages])
        return next(self.responses)


def test_load_attacks_from_json_file() -> None:
    attacks = load_attacks(ATTACKS_FILE)

    assert attacks
    assert all(isinstance(attack, AttackCase) for attack in attacks)
    assert attacks[0].id == "PI-DIRECT-001"
    assert attacks[0].category == "pi.direct"
    assert attacks[0].messages[0]["role"] == "user"
    assert attacks[0].evaluator == {
        "type": "contains",
        "value": "REDTEAM-7392",
    }
    assert attacks[0].turns is None

    multi_turn_attack = attacks[-1]
    assert multi_turn_attack.id == "JB-MULTITURN-001"
    assert multi_turn_attack.messages == []
    assert len(multi_turn_attack.turns) == 2


def test_attack_case_rejects_messages_and_turns_together() -> None:
    data = {
        "id": "conflicting-attack",
        "name": "Conflicting attack",
        "category": "test.conflict",
        "description": "An invalid attack case.",
        "system_prompt": "System instruction",
        "messages": [{"role": "user", "content": "Existing message"}],
        "turns": ["First turn"],
        "evaluator": {"type": "contains", "value": "canary"},
    }

    with pytest.raises(
        ValueError,
        match="cannot define both non-empty 'messages' and 'turns'",
    ):
        AttackCase.from_dict(data)


def test_run_attack_preserves_single_turn_behavior() -> None:
    attack = AttackCase(
        id="single-turn",
        name="Single turn",
        category="test.single",
        description="A single-turn test case.",
        system_prompt="System instruction",
        messages=[{"role": "user", "content": "User prompt"}],
        evaluator={"type": "contains", "value": "canary"},
    )
    target = RecordingTarget(["Final response with canary"])

    result = run_attack(
        attack=attack,
        target=target,
        run_id="run-123",
        trial_number=1,
        git_commit="abc1234",
    )

    assert target.histories == [
        [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "User prompt"},
        ]
    ]
    assert result.response == "Final response with canary"
    assert result.attack_success is True


def test_run_attack_builds_history_and_evaluates_only_final_response() -> None:
    attack = AttackCase(
        id="multi-turn",
        name="Multi-turn",
        category="test.multiturn",
        description="A multi-turn test case.",
        system_prompt="System instruction",
        messages=[],
        evaluator={"type": "contains", "value": "canary"},
        turns=["First turn", "Second turn"],
    )
    target = RecordingTarget(
        [
            "Intermediate response with canary",
            "Final response without the evaluated value",
        ]
    )

    result = run_attack(
        attack=attack,
        target=target,
        run_id="run-123",
        trial_number=1,
        git_commit="abc1234",
    )

    assert target.histories == [
        [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "First turn"},
        ],
        [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "First turn"},
            {
                "role": "assistant",
                "content": "Intermediate response with canary",
            },
            {"role": "user", "content": "Second turn"},
        ],
    ]
    assert result.response == "Final response without the evaluated value"
    assert result.attack_success is False
