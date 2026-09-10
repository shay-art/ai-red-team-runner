from targets.ollama import OllamaTarget


def test_normalize_response_strips_whitespace() -> None:
    target = OllamaTarget(model="test-model")

    assert target.normalize_response("  model response\n") == "model response"


def test_normalize_response_removes_thinking_content() -> None:
    target = OllamaTarget(model="test-model")

    response = "private reasoning</think>\n  final response  "

    assert target.normalize_response(response) == "final response"
