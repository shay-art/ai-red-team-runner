from models import AttackCase
from runner import ATTACKS_FILE, load_attacks


def test_load_attacks_from_json_file() -> None:
    attacks = load_attacks(ATTACKS_FILE)

    assert attacks
    assert all(isinstance(attack, AttackCase) for attack in attacks)
    assert attacks[0].id == "PI-DIRECT-001"
    assert attacks[0].category == "pi.direct"
    assert attacks[0].messages[0]["role"] == "user"
    assert attacks[0].evaluator == {
        "type": "contains",
        "value": "REDTEAM-7392",
    }
