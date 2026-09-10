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

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            description=data["description"],
            system_prompt=data["system_prompt"],
            messages=data["messages"],
            evaluator=data["evaluator"],
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