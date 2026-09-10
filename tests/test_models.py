from models import AttackResult


def test_attack_result_serialization() -> None:
    result = AttackResult(
        run_id="run-123",
        trial_number=2,
        attack_id="PI-DIRECT-001",
        category="pi.direct",
        model="phi3:mini",
        response="model response",
        attack_success=False,
        timestamp="2026-09-10T12:00:00+00:00",
        git_commit="a09ac00",
    )

    assert result.to_dict() == {
        "run_id": "run-123",
        "trial_number": 2,
        "attack_id": "PI-DIRECT-001",
        "category": "pi.direct",
        "model": "phi3:mini",
        "response": "model response",
        "attack_success": False,
        "timestamp": "2026-09-10T12:00:00+00:00",
        "git_commit": "a09ac00",
    }
