from evaluators.rules import contains


def test_contains_is_case_insensitive() -> None:
    assert contains("The protected token is REDTEAM-7392.", "redteam-7392")


def test_contains_returns_false_when_value_is_absent() -> None:
    assert not contains("The response does not include the canary.", "REDTEAM-7392")
