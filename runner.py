import json
from datetime import datetime, timezone

from targets.ollama import OllamaTarget
from evaluators.rules import contains


with open("attacks/attacks.json", "r", encoding="utf-8") as file:
    attacks = json.load(file)

attack = attacks[0]

messages = [
    {
        "role": "system",
        "content": attack["system_prompt"],
    },
    *attack["messages"],
]

model_name = "qwen3:4b"
target = OllamaTarget(model=model_name)

response = target.send(messages)

evaluator = attack["evaluator"]

if evaluator["type"] == "contains":
    attack_success = contains(
        response=response,
        value=evaluator["value"],
    )
else:
    raise ValueError(f"Unsupported evaluator: {evaluator['type']}")

result = {
    "attack_id": attack["id"],
    "category": attack["category"],
    "model": model_name,
    "response": response,
    "attack_success": attack_success,
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

with open("results/results.jsonl", "a", encoding="utf-8") as file:
    file.write(json.dumps(result) + "\n")

print("ATTACK ID:", attack["id"])
print("CATEGORY:", attack["category"])
print("MODEL:", model_name)
print("MODEL RESPONSE:")
print(response)
print()
print("ATTACK SUCCESS:", attack_success)
print("RESULT SAVED: results/results.jsonl")