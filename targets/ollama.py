import requests


class OllamaTarget:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def normalize_response(self, content: str) -> str:
        if "</think>" in content:
            content = content.split("</think>", 1)[1]

        return content.strip()

    def send(self, messages: list[dict]) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        raw_content = data["message"]["content"]

        return self.normalize_response(raw_content)