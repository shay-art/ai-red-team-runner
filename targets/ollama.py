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

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        prepared = [message.copy() for message in messages]

        for index in range(len(prepared) - 1, -1, -1):
            if prepared[index]["role"] == "user":
                prepared[index]["content"] = (
                    prepared[index]["content"].rstrip()
                    + "\n\n/no_think"
                )
                break

        return prepared

    def send(self, messages: list[dict]) -> str:
        prepared_messages = self.prepare_messages(messages)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": prepared_messages,
                "stream": False,
                "think": False,
                "options": {
                    "num_predict": 256,
                    "temperature": 0,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()
        raw_content = data["message"]["content"]

        return self.normalize_response(raw_content)