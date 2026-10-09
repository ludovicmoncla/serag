from pathlib import Path

import pandas as pd

from serad_kg.data import Vocabulary
from serad_kg.description import dataset_summary, readable_events, validate_events


def test_readable_events_and_summary() -> None:
    entities = Vocabulary({"Alice": 0, "Bob": 1}, {0: "Alice", 1: "Bob"})
    relations = Vocabulary({"knows": 0}, {0: "knows"})
    events = pd.DataFrame({"subject": [0], "relation": [0], "object": [1], "time": [3]})

    validate_events(events, entities, relations, Path("example.txt"))
    readable = readable_events(events, entities, relations)
    summary = dataset_summary({"train": events}, entities, relations)

    assert readable.loc[0, ["subject", "relation", "object"]].tolist() == [
        "Alice",
        "knows",
        "Bob",
    ]
    assert summary["num_events"] == 1
    assert summary["time_min"] == 3
