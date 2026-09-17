from dataclasses import asdict, dataclass


SUPPORTED_MESSAGE_ROLES = {"assistant", "user"}


@dataclass
class AttackCase:
    id: str
    name: str
    category: str
    description: str
    system_prompt: str
    messages: list[dict]
    evaluator: dict
    turns: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict):
        messages = data.get("messages", [])
        turns = data.get("turns")

        if messages and turns:
            raise ValueError(
                "AttackCase cannot define both non-empty 'messages' and 'turns'"
            )

        if messages is None:
            messages = []

        if not isinstance(messages, list):
            raise ValueError("AttackCase 'messages' must be a list")

        if turns is not None and not isinstance(turns, list):
            raise ValueError("AttackCase 'turns' must be a list or null")

        if not messages and not turns:
            raise ValueError("AttackCase must define a non-empty conversation")

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("AttackCase messages must be objects")

            role = message.get("role")

            if role not in SUPPORTED_MESSAGE_ROLES:
                raise ValueError(
                    f"Unsupported AttackCase message role: {role!r}"
                )

        if "evaluator" not in data:
            raise ValueError("AttackCase must define an evaluator")

        evaluator = data["evaluator"]

        if not isinstance(evaluator, dict):
            raise ValueError("AttackCase evaluator must be an object")

        evaluator_type = evaluator.get("type")

        if evaluator_type != "contains":
            raise ValueError(f"Unsupported evaluator: {evaluator_type!r}")

        if "value" not in evaluator:
            raise ValueError("AttackCase evaluator must define a value")

        evaluator_value = evaluator["value"]

        if not isinstance(evaluator_value, str):
            raise ValueError("AttackCase evaluator value must be a string")

        if not evaluator_value.strip():
            raise ValueError("AttackCase evaluator value must not be empty")

        return cls(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            description=data["description"],
            system_prompt=data["system_prompt"],
            messages=messages,
            evaluator=evaluator,
            turns=turns,
        )


@dataclass
class AttackResult:
    run_id: str
    trial_number: int
    attack_id: str
    category: str
    model: str
    response: str
    attack_success: bool
    timestamp: str
    git_commit: str
    responses: list[str] | None = None
    git_commit_full: str = "unknown"
    git_dirty: bool | None = None
    dataset_hash: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)
