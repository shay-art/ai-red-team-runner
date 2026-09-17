import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from evaluators.rules import contains
from models import AttackCase, AttackResult
from targets.ollama import OllamaTarget


PROJECT_ROOT = Path(__file__).resolve().parent
ATTACKS_FILE = PROJECT_ROOT / "attacks" / "attacks.json"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_NAME = "phi3:mini"
TRIALS_PER_ATTACK = 3


def load_attacks(path: Path) -> list[AttackCase]:
    with open(path, "r", encoding="utf-8") as file:
        attacks_data = json.load(file)

    attacks = [AttackCase.from_dict(data) for data in attacks_data]
    seen_ids = set()

    for attack in attacks:
        if attack.id in seen_ids:
            raise ValueError(f"Duplicate attack ID: {attack.id}")

        seen_ids.add(attack.id)

    return attacks


def evaluate_response(attack: AttackCase, response: str) -> bool:
    evaluator = attack.evaluator

    if evaluator["type"] == "contains":
        return contains(
            response=response,
            value=evaluator["value"],
        )

    raise ValueError(f"Unsupported evaluator: {evaluator['type']}")


def get_git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_full_git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_dirty(project_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )

        return bool(result.stdout.strip())

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_run_id(git_commit: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:6]

    return f"{timestamp}_{git_commit}_{suffix}"


def create_results_file(
    results_dir: Path,
    run_id: str,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"{run_id}.jsonl"

    results_file.touch(exist_ok=False)

    return results_file


def save_result(
    result: AttackResult,
    path: Path,
) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(result.to_dict()) + "\n")


def run_attack(
    attack: AttackCase,
    target: OllamaTarget,
    run_id: str,
    trial_number: int,
    git_commit: str,
    git_commit_full: str = "unknown",
    git_dirty: bool | None = None,
    dataset_hash: str = "unknown",
) -> AttackResult:
    messages = [
        {
            "role": "system",
            "content": attack.system_prompt,
        },
        *attack.messages,
    ]

    responses = None

    if attack.turns:
        responses = []

        for turn in attack.turns:
            messages.append(
                {
                    "role": "user",
                    "content": turn,
                }
            )

            response = target.send(messages)
            responses.append(response)

            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )
    else:
        response = target.send(messages)

    if responses is not None:
        attack_success = any(
            evaluate_response(
                attack=attack,
                response=assistant_response,
            )
            for assistant_response in responses
        )
    else:
        attack_success = evaluate_response(
            attack=attack,
            response=response,
        )

    return AttackResult(
        run_id=run_id,
        trial_number=trial_number,
        attack_id=attack.id,
        category=attack.category,
        model=target.model,
        response=response,
        attack_success=attack_success,
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=git_commit,
        responses=responses,
        git_commit_full=git_commit_full,
        git_dirty=git_dirty,
        dataset_hash=dataset_hash,
    )


def validate_run_completion(
    results: list[AttackResult],
    attacks: list[AttackCase],
    trials_per_attack: int,
) -> None:
    expected_attack_ids = {attack.id for attack in attacks}
    expected_trial_numbers = set(range(1, trials_per_attack + 1))
    expected_total = len(attacks) * trials_per_attack
    actual_attack_ids = {result.attack_id for result in results}

    unknown_attack_ids = actual_attack_ids - expected_attack_ids

    if unknown_attack_ids:
        raise ValueError(
            "Results contain unknown attack IDs: "
            f"{sorted(unknown_attack_ids)}"
        )

    if len(results) != expected_total:
        raise ValueError(
            "Incomplete benchmark: expected "
            f"{expected_total} results, got {len(results)}"
        )

    results_by_attack = defaultdict(list)

    for result in results:
        results_by_attack[result.attack_id].append(result.trial_number)

    for attack in attacks:
        trial_numbers = results_by_attack[attack.id]

        if len(trial_numbers) != len(set(trial_numbers)):
            raise ValueError(
                f"Duplicate trial numbers for attack {attack.id}"
            )

        if set(trial_numbers) != expected_trial_numbers:
            raise ValueError(
                f"Invalid trial numbers for attack {attack.id}: "
                f"expected {sorted(expected_trial_numbers)}, "
                f"got {sorted(trial_numbers)}"
            )


def print_trial_result(
    result: AttackResult,
    trial_number: int,
    total_trials: int,
) -> None:
    print("-" * 60)
    print(
        f"{result.attack_id} | "
        f"TRIAL {trial_number}/{total_trials} | "
        f"SUCCESS: {result.attack_success}"
    )


def print_summary(results: list[AttackResult]) -> None:
    total = len(results)

    if not total:
        print()
        print("=" * 60)
        print("RUN SUMMARY")
        print("=" * 60)
        print("No executions; ASR is not available.")
        return

    successful = sum(result.attack_success for result in results)

    overall_asr = successful / total * 100

    print()
    print("=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"Total executions:   {total}")
    print(f"Successful attacks: {successful}")
    print(f"Overall ASR:        {overall_asr:.1f}%")

    attack_stats = defaultdict(
        lambda: {
            "category": "",
            "total": 0,
            "successful": 0,
        }
    )

    for result in results:
        stats = attack_stats[result.attack_id]

        stats["category"] = result.category
        stats["total"] += 1

        if result.attack_success:
            stats["successful"] += 1

    print()
    print("ASR BY ATTACK")

    for attack_id, stats in sorted(attack_stats.items()):
        attack_total = stats["total"]
        attack_successful = stats["successful"]

        attack_asr = (
            attack_successful / attack_total * 100
            if attack_total
            else 0.0
        )

        print(
            f"{attack_id:<20} "
            f"{attack_successful}/{attack_total} "
            f"({attack_asr:.1f}%)"
        )

    category_stats = defaultdict(
        lambda: {
            "total": 0,
            "successful": 0,
        }
    )

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

    git_commit = get_git_commit(PROJECT_ROOT)
    git_commit_full = get_full_git_commit(PROJECT_ROOT)
    git_dirty = get_git_dirty(PROJECT_ROOT)
    dataset_hash = get_dataset_hash(ATTACKS_FILE)
    run_id = create_run_id(git_commit)

    results_file = create_results_file(
        results_dir=RESULTS_DIR,
        run_id=run_id,
    )

    results = []

    print(f"RUN ID: {run_id}")
    print(f"GIT COMMIT: {git_commit}")
    print(f"FULL GIT COMMIT: {git_commit_full}")
    print(f"GIT DIRTY: {git_dirty}")
    print(f"DATASET SHA256: {dataset_hash}")
    print(f"LOADED ATTACKS: {len(attacks)}")
    print(f"TRIALS PER ATTACK: {TRIALS_PER_ATTACK}")
    print(f"TARGET MODEL: {target.model}")
    print(f"RESULTS FILE: {results_file}")
    print()

    for attack in attacks:
        print("=" * 60)
        print(f"RUNNING: {attack.id}")
        print(f"CATEGORY: {attack.category}")

        for trial_number in range(1, TRIALS_PER_ATTACK + 1):
            result = run_attack(
                attack=attack,
                target=target,
                run_id=run_id,
                trial_number=trial_number,
                git_commit=git_commit,
                git_commit_full=git_commit_full,
                git_dirty=git_dirty,
                dataset_hash=dataset_hash,
            )

            save_result(
                result=result,
                path=results_file,
            )

            results.append(result)

            print_trial_result(
                result=result,
                trial_number=trial_number,
                total_trials=TRIALS_PER_ATTACK,
            )

    validate_run_completion(
        results=results,
        attacks=attacks,
        trials_per_attack=TRIALS_PER_ATTACK,
    )

    print_summary(results)

    print()
    print(f"RUN ID: {run_id}")
    print(f"RESULTS SAVED: {results_file}")


if __name__ == "__main__":
    main()
