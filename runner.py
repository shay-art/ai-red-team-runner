import json
from datetime import datetime, timezone

from evaluators.rules import contains
from models import AttackCase, AttackResult
from targets.ollama import OllamaTarget


MODEL_NAME = "qwen3:4b"
ATTACKS_FILE = "attacks/attacks.json"
RESULTS_FILE = "results/results.jsonl"


def load_attacks(path: str) -> list[AttackCase]:
    with open(path, "r", encoding="utf-8") as file:
        attacks_data = json.load(file)

    return [AttackCase.from_dict(data) for data in attacks_data]


def evaluate_response(attack: AttackCase, response: str) -> bool:
    evaluator = attack.evaluator

    if evaluator["type"] == "contains":
        return contains(
            response=response,
            value=evaluator["value"],
        )

    raise ValueError(f"Unsupported evaluator: {evaluator['type']}")


def save_result(result: AttackResult, path: str) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(result.to_dict()) + "\n")


def run_attack(
    attack: AttackCase,
    target: OllamaTarget,
    model_name: str,
) -> AttackResult:
    messages = [
        {
            "role": "system",
            "content": attack.system_prompt,
        },
        *attack.messages,
    ]

    response = target.send(messages)

    attack_success = evaluate_response(
        attack=attack,
        response=response,
    )

    return AttackResult(
        attack_id=attack.id,
        category=attack.category,
        model=model_name,
        response=response,
        attack_success=attack_success,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def main() -> None:
    attacks = load_attacks(ATTACKS_FILE)

    attack = attacks[0]

    target = OllamaTarget(model=MODEL_NAME)

    result = run_attack(
        attack=attack,
        target=target,
        model_name=MODEL_NAME,
    )

    save_result(
        result=result,
        path=RESULTS_FILE,
    )

    print("ATTACK ID:", result.attack_id)
    print("CATEGORY:", result.category)
    print("MODEL:", result.model)
    print("MODEL RESPONSE:")
    print(result.response)
    print()
    print("ATTACK SUCCESS:", result.attack_success)
    print(f"RESULT SAVED: {RESULTS_FILE}")


if __name__ == "__main__":
    main()