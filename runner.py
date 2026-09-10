from targets.ollama import OllamaTarget


target = OllamaTarget(model="qwen3:4b")

messages = [
    {
        "role": "user",
        "content": "Reply with exactly: RED TEAM RUNNER READY"
    }
]

response = target.send(messages)

print("MODEL RESPONSE:")
print(response)