from serad_kg.model import SeradKG, Serag


def test_model_uses_canonical_name() -> None:
    assert Serag.__name__ == "Serag"


def test_legacy_model_name_is_an_alias() -> None:
    assert SeradKG is Serag
