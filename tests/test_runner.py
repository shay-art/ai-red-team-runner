import json

import pytest

from models import AttackCase, AttackResult
from runner import (
    ATTACKS_FILE,
    get_dataset_hash,
    load_attacks,
    print_summary,
    run_attack,
    validate_run_completion,
)


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

    multi_turn_attack = next(
        attack for attack in attacks if attack.id == "JB-MULTITURN-001"
    )
    assert multi_turn_attack.id == "JB-MULTITURN-001"
    assert multi_turn_attack.messages == []
    assert len(multi_turn_attack.turns) == 2


def test_load_attacks_rejects_duplicate_ids(tmp_path) -> None:
    attack = {
        "id": "duplicate-attack",
        "name": "Duplicate attack",
        "category": "test.duplicate",
        "description": "A duplicated attack case.",
        "system_prompt": "System instruction",
        "messages": [{"role": "user", "content": "User prompt"}],
        "evaluator": {"type": "contains", "value": "canary"},
    }
    attacks_file = tmp_path / "attacks.json"
    attacks_file.write_text(json.dumps([attack, attack]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate attack ID: duplicate-attack"):
        load_attacks(attacks_file)


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


@pytest.mark.parametrize(
    ("conversation", "expected_error"),
    [
        ({}, "must define a non-empty conversation"),
        ({"messages": []}, "must define a non-empty conversation"),
        ({"turns": []}, "must define a non-empty conversation"),
    ],
)
def test_attack_case_rejects_empty_conversations(
    conversation: dict,
    expected_error: str,
) -> None:
    data = {
        "id": "empty-attack",
        "name": "Empty attack",
        "category": "test.empty",
        "description": "An invalid empty attack case.",
        "system_prompt": "System instruction",
        "evaluator": {"type": "contains", "value": "canary"},
        **conversation,
    }

    with pytest.raises(ValueError, match=expected_error):
        AttackCase.from_dict(data)


@pytest.mark.parametrize("role", [None, "system", "tool", "invalid"])
def test_attack_case_rejects_unsupported_message_roles(role) -> None:
    message = {"content": "User prompt"}

    if role is not None:
        message["role"] = role

    data = {
        "id": "invalid-role-attack",
        "name": "Invalid role attack",
        "category": "test.invalid-role",
        "description": "An attack case with an invalid message role.",
        "system_prompt": "System instruction",
        "messages": [message],
        "evaluator": {"type": "contains", "value": "canary"},
    }

    with pytest.raises(ValueError, match="Unsupported AttackCase message role"):
        AttackCase.from_dict(data)


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_attack_case_accepts_supported_message_roles(role: str) -> None:
    data = {
        "id": "supported-role-attack",
        "name": "Supported role attack",
        "category": "test.supported-role",
        "description": "An attack case with a supported message role.",
        "system_prompt": "System instruction",
        "messages": [{"role": role, "content": "Conversation content"}],
        "evaluator": {"type": "contains", "value": "canary"},
    }

    attack = AttackCase.from_dict(data)

    assert attack.messages[0]["role"] == role


def valid_attack_data() -> dict:
    return {
        "id": "valid-attack",
        "name": "Valid attack",
        "category": "test.valid",
        "description": "A valid attack case.",
        "system_prompt": "System instruction",
        "messages": [{"role": "user", "content": "User prompt"}],
        "evaluator": {"type": "contains", "value": "canary"},
    }


def test_attack_case_rejects_missing_evaluator() -> None:
    data = valid_attack_data()
    del data["evaluator"]

    with pytest.raises(ValueError, match="must define an evaluator"):
        AttackCase.from_dict(data)


def test_attack_case_rejects_missing_evaluator_value() -> None:
    data = valid_attack_data()
    del data["evaluator"]["value"]

    with pytest.raises(ValueError, match="evaluator must define a value"):
        AttackCase.from_dict(data)


@pytest.mark.parametrize("value", ["", "   "])
def test_attack_case_rejects_empty_evaluator_value(value: str) -> None:
    data = valid_attack_data()
    data["evaluator"]["value"] = value

    with pytest.raises(ValueError, match="value must not be empty"):
        AttackCase.from_dict(data)


@pytest.mark.parametrize("value", [None, 123, ["canary"]])
def test_attack_case_rejects_non_string_evaluator_value(value) -> None:
    data = valid_attack_data()
    data["evaluator"]["value"] = value

    with pytest.raises(ValueError, match="value must be a string"):
        AttackCase.from_dict(data)


def test_attack_case_rejects_unsupported_evaluator_type() -> None:
    data = valid_attack_data()
    data["evaluator"]["type"] = "semantic"

    with pytest.raises(ValueError, match="Unsupported evaluator"):
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
        git_commit_full="abc1234567890",
        git_dirty=True,
        dataset_hash="dataset-sha256",
    )

    assert target.histories == [
        [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "User prompt"},
        ]
    ]
    assert result.response == "Final response with canary"
    assert result.responses is None
    assert result.attack_success is True
    assert result.git_commit == "abc1234"
    assert result.git_commit_full == "abc1234567890"
    assert result.git_dirty is True
    assert result.dataset_hash == "dataset-sha256"


def test_run_attack_counts_early_multi_turn_leak_as_success() -> None:
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
    assert result.responses == [
        "Intermediate response with canary",
        "Final response without the evaluated value",
    ]
    assert result.attack_success is True


@pytest.mark.parametrize(
    ("responses", "expected_success"),
    [
        (["Safe intermediate response", "Final response with canary"], True),
        (["Safe intermediate response", "Safe final response"], False),
    ],
)
def test_run_attack_evaluates_all_multi_turn_responses(
    responses: list[str],
    expected_success: bool,
) -> None:
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
    target = RecordingTarget(responses)

    result = run_attack(
        attack=attack,
        target=target,
        run_id="run-123",
        trial_number=1,
        git_commit="abc1234",
    )

    assert result.responses == responses
    assert result.response == responses[-1]
    assert result.attack_success is expected_success


def make_result(
    attack_id: str,
    category: str,
    trial_number: int,
    attack_success: bool = False,
) -> AttackResult:
    return AttackResult(
        run_id="run-123",
        trial_number=trial_number,
        attack_id=attack_id,
        category=category,
        model="test-model",
        response="response",
        attack_success=attack_success,
        timestamp="2026-09-17T00:00:00+00:00",
        git_commit="abc1234",
    )


def make_attack(attack_id: str, category: str) -> AttackCase:
    return AttackCase(
        id=attack_id,
        name="Test attack",
        category=category,
        description="A test attack.",
        system_prompt="System instruction",
        messages=[{"role": "user", "content": "User prompt"}],
        evaluator={"type": "contains", "value": "canary"},
    )


def test_validate_run_completion_accepts_expected_trials() -> None:
    attacks = [make_attack("attack-1", "category-1")]
    results = [
        make_result("attack-1", "category-1", 1),
        make_result("attack-1", "category-1", 2),
        make_result("attack-1", "category-1", 3),
    ]

    validate_run_completion(results, attacks, trials_per_attack=3)


def test_v2_dataset_has_expected_complete_execution_count() -> None:
    attacks = load_attacks(ATTACKS_FILE)
    results = [
        make_result(attack.id, attack.category, trial_number)
        for attack in attacks
        for trial_number in range(1, 4)
    ]

    assert len(attacks) == 29
    assert len(results) == 87
    validate_run_completion(results, attacks, trials_per_attack=3)


def test_validate_run_completion_rejects_missing_trials() -> None:
    attacks = [make_attack("attack-1", "category-1")]
    results = [
        make_result("attack-1", "category-1", 1),
        make_result("attack-1", "category-1", 2),
    ]

    with pytest.raises(ValueError, match="Incomplete benchmark"):
        validate_run_completion(results, attacks, trials_per_attack=3)


def test_validate_run_completion_rejects_duplicate_trial_numbers() -> None:
    attacks = [make_attack("attack-1", "category-1")]
    results = [
        make_result("attack-1", "category-1", 1),
        make_result("attack-1", "category-1", 1),
        make_result("attack-1", "category-1", 3),
    ]

    with pytest.raises(ValueError, match="Duplicate trial numbers"):
        validate_run_completion(results, attacks, trials_per_attack=3)


def test_print_summary_reports_no_executions(capsys) -> None:
    print_summary([])

    output = capsys.readouterr().out

    assert "No executions; ASR is not available." in output
    assert "Overall ASR" not in output


def test_print_summary_calculates_mixed_asr(capsys) -> None:
    results = [
        make_result("attack-1", "category-1", 1, True),
        make_result("attack-1", "category-1", 2, False),
        make_result("attack-2", "category-2", 1, True),
        make_result("attack-2", "category-2", 2, True),
    ]

    print_summary(results)

    output = capsys.readouterr().out

    assert "Successful attacks: 3" in output
    assert "Overall ASR:        75.0%" in output
    assert "attack-1             1/2 (50.0%)" in output
    assert "attack-2             2/2 (100.0%)" in output
    assert "category-1      1/2 (50.0%)" in output
    assert "category-2      2/2 (100.0%)" in output


def test_dataset_hash_changes_with_content(tmp_path) -> None:
    dataset = tmp_path / "attacks.json"
    dataset.write_text("first dataset", encoding="utf-8")
    first_hash = get_dataset_hash(dataset)

    dataset.write_text("second dataset", encoding="utf-8")
    second_hash = get_dataset_hash(dataset)

    assert first_hash != second_hash
