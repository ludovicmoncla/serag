from pathlib import Path

from serad_kg.data import load_events, load_vocabulary


def test_load_repository_data() -> None:
    data_dir = Path("data/icews18")
    entities = load_vocabulary(data_dir / "entity2id.txt")
    relations = load_vocabulary(data_dir / "relation2id.txt")
    events = load_events(data_dir / "train.txt")

    assert len(entities.id_to_text) == 23033
    assert len(relations.id_to_text) == 256
    assert list(events.columns) == ["subject", "relation", "object", "time"]
    assert len(events) > 100_000
