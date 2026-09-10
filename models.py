from dataclasses import asdict, dataclass


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

        return cls(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            description=data["description"],
            system_prompt=data["system_prompt"],
            messages=messages,
            evaluator=data["evaluator"],
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

    def to_dict(self) -> dict:
        return asdict(self)
