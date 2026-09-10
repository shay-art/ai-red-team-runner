from dataclasses import dataclass


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
    attack_id: str
    category: str
    model: str
    response: str
    attack_success: bool
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "attack_id": self.attack_id,
            "category": self.category,
            "model": self.model,
            "response": self.response,
            "attack_success": self.attack_success,
            "timestamp": self.timestamp,
        }