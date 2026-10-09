import json
from types import SimpleNamespace

import numpy as np
import pytest

from serad_kg.anomalies import (
    corrupt_triples,
    generate_anomalies_genai,
    parse_generated_anomalies,
    validate_generated_anomalies,
)


def test_corrupt_triples_changes_exactly_one_component() -> None:
    triples = np.array([[0, 0, 1], [1, 1, 2], [2, 0, 0]])
    anomalies = corrupt_triples(triples, 3, 3, 2, np.random.default_rng(42))

    assert anomalies.shape == (3, 3)
    assert not {tuple(row) for row in anomalies} & {tuple(row) for row in triples}
    assert all(
        any(
            np.array_equal(row[:column], source[:column])
            and np.array_equal(row[column + 1 :], source[column + 1 :])
            and row[column] != source[column]
            for source in triples
            for column in range(3)
        )
        for row in anomalies
    )


def test_parse_generated_anomalies() -> None:
    content = json.dumps(
        {"anomalies": [{"subject": "France", "relation": "invades", "object": "France"}]}
    )
    assert parse_generated_anomalies(content, 1) == [("France", "invades", "France")]


def test_parse_generated_anomalies_rejects_duplicates() -> None:
    item = {"subject": "France", "relation": "invades", "object": "France"}
    with pytest.raises(ValueError, match="duplicate"):
        parse_generated_anomalies(json.dumps({"anomalies": [item, item]}), 2)


def test_validate_generated_anomalies_requires_a_context_entity() -> None:
    with pytest.raises(ValueError, match="reuse an entity"):
        validate_generated_anomalies(
            [("Mars", "signs treaty with", "Venus")],
            [("France", "meets", "Germany")],
        )


def test_validate_generated_anomalies_can_require_only_context_labels() -> None:
    context = [("France", "meets", "Germany")]
    validate_generated_anomalies(
        [("Germany", "meets", "France")],
        context,
        require_context_vocabulary=True,
    )
    with pytest.raises(ValueError, match="outside the snapshot context"):
        validate_generated_anomalies(
            [("Germany", "invades", "France")],
            context,
            require_context_vocabulary=True,
        )


def test_strict_genai_schema_uses_safe_tokens_for_unsafe_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            content = json.dumps(
                {"anomalies": [{"subject": "E1", "relation": "R0", "object": "E0"}]}
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    anomalies = generate_anomalies_genai(
        1,
        [("Back\\slash", "Meet", "Other")],
        api_key="test-key",
        model="test-model",
        seed=42,
        require_context_vocabulary=True,
    )

    assert anomalies == [("Other", "Meet", "Back\\slash")]
    schema = captured["response_format"]["json_schema"]["schema"]  # type: ignore[index]
    properties = schema["properties"]["anomalies"]["items"]["properties"]
    assert properties["subject"]["enum"] == ["E0", "E1"]
    assert all("\\" not in token for token in properties["subject"]["enum"])
