import json
from collections import defaultdict
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


def print_result(result: AttackResult) -> None:
    print("=" * 60)
    print("ATTACK ID:", result.attack_id)
    print("CATEGORY:", result.category)
    print("MODEL:", result.model)
    print("MODEL RESPONSE:")
    print(result.response)
    print()
    print("ATTACK SUCCESS:", result.attack_success)


def print_summary(results: list[AttackResult]) -> None:
    total = len(results)
    successful = sum(result.attack_success for result in results)

    overall_asr = (successful / total * 100) if total else 0.0

    print()
    print("=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"Total attacks:      {total}")
    print(f"Successful attacks: {successful}")
    print(f"Overall ASR:        {overall_asr:.1f}%")

    category_stats = defaultdict(lambda: {"total": 0, "successful": 0})

    for result in results:
        category_stats[result.category]["total"] += 1

        if result.attack_success:
            category_stats[result.category]["successful"] += 1

    print()
    print("ASR BY CATEGORY")

    for category, stats in sorted(category_stats.items()):
        category_total = stats["total"]
        category_successful = stats["successful"]

        category_asr = (
            category_successful / category_total * 100
            if category_total
            else 0.0
        )

        print(
            f"{category:<15} "
            f"{category_successful}/{category_total} "
            f"({category_asr:.1f}%)"
        )


def main() -> None:
    attacks = load_attacks(ATTACKS_FILE)

    target = OllamaTarget(model=MODEL_NAME)

    results = []

    print(f"LOADED ATTACKS: {len(attacks)}")
    print(f"TARGET MODEL: {MODEL_NAME}")
    print()

    for attack in attacks:
        result = run_attack(
            attack=attack,
            target=target,
            model_name=MODEL_NAME,
        )

        save_result(
            result=result,
            path=RESULTS_FILE,
        )

        results.append(result)

        print_result(result)

    print_summary(results)

    print()
    print(f"RESULTS SAVED: {RESULTS_FILE}")


if __name__ == "__main__":
    main()