from unittest.mock import Mock, patch

from targets.ollama import OllamaTarget


def test_normalize_response_strips_whitespace() -> None:
    target = OllamaTarget(model="test-model")

    assert target.normalize_response("  model response\n") == "model response"


def test_normalize_response_removes_thinking_content() -> None:
    target = OllamaTarget(model="test-model")

    response = "private reasoning</think>\n  final response  "

    assert target.normalize_response(response) == "final response"


def test_send_accepts_single_prompt_and_normalizes_response() -> None:
    target = OllamaTarget(model="test-model")
    response = Mock()
    response.json.return_value = {
        "message": {
            "content": "private reasoning</think>\n  final response  ",
        }
    }

    with patch("targets.ollama.requests.post", return_value=response) as post:
        result = target.send("single prompt")

    assert result == "final response"
    assert post.call_args.kwargs["json"]["messages"] == [
        {
            "role": "user",
            "content": "single prompt\n\n/no_think",
        }
    ]


def test_send_preserves_message_history_and_normalizes_response() -> None:
    target = OllamaTarget(model="test-model")
    messages = [
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "first response"},
        {"role": "user", "content": "follow-up prompt"},
    ]
    response = Mock()
    response.json.return_value = {
        "message": {
            "content": "  follow-up response  ",
        }
    }

    with patch("targets.ollama.requests.post", return_value=response) as post:
        result = target.send(messages)

    assert result == "follow-up response"
    assert post.call_args.kwargs["json"]["messages"] == [
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "first response"},
        {
            "role": "user",
            "content": "follow-up prompt\n\n/no_think",
        },
    ]
    assert messages[-1]["content"] == "follow-up prompt"
