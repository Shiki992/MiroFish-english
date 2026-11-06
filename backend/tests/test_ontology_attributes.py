from app.services.ontology_generator import OntologyGenerator
from app.services.graph_builder import GraphBuilderService
from app.utils.ontology import (
    MAX_ONTOLOGY_ATTRIBUTES,
    MAX_ONTOLOGY_SOURCE_TARGETS,
    normalize_ontology_attribute,
    normalize_ontology_attributes,
)
from app.utils.graph_client import build_ontology_models


def test_normalize_string_attribute():
    assert normalize_ontology_attribute("role") == {
        "name": "role",
        "type": "text",
        "description": "role",
    }


def test_preserve_valid_dictionary_attribute():
    original = {"name": "role", "type": "text", "description": "Public role"}
    assert normalize_ontology_attribute(original) == original
    assert normalize_ontology_attribute(original) is not original


def test_reject_unusable_attribute_shapes():
    for value in (None, 7, [], {}, {"name": None}, {"name": ""}, "   "):
        assert normalize_ontology_attribute(value) is None


def test_attribute_list_is_non_empty_and_capped_for_zep():
    assert normalize_ontology_attributes(None) == [{
        "name": "details",
        "type": "text",
        "description": "Additional details about this ontology type.",
    }]

    attributes = [None] + [f"field_{index}" for index in range(12)]
    normalized = normalize_ontology_attributes(attributes)

    assert len(normalized) == MAX_ONTOLOGY_ATTRIBUTES
    assert [attribute["name"] for attribute in normalized] == [
        f"field_{index}" for index in range(MAX_ONTOLOGY_ATTRIBUTES)
    ]


def test_generator_normalizes_entity_and_edge_attributes():
    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "speaker", "attributes": ["role", None]}],
        "edge_types": [{"name": "quotes", "attributes": ["source_url", {}]}],
    })

    assert result["entity_types"][0]["attributes"] == [{
        "name": "role",
        "type": "text",
        "description": "role",
    }]
    assert result["edge_types"][0]["attributes"] == [{
        "name": "source_url",
        "type": "text",
        "description": "source_url",
    }]


def test_generator_adds_a_property_to_empty_custom_types():
    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "speaker", "attributes": []}],
        "edge_types": [{"name": "quotes", "attributes": []}],
    })

    assert result["entity_types"][0]["attributes"][0]["name"] == "details"
    assert result["edge_types"][0]["attributes"][0]["name"] == "details"


def test_graph_builder_safety_net_accepts_strings_and_skips_invalid_values():
    captured = {}

    class GraphApi:
        def set_ontology(self, **kwargs):
            captured.update(kwargs)

    class Client:
        graph = GraphApi()

    builder = object.__new__(GraphBuilderService)
    builder.client = Client()
    builder.set_ontology("graph-id", {
        "entity_types": [{
            "name": "Speaker",
            "attributes": ["role", None, {"name": "summary"}],
        }],
        "edge_types": [],
    })

    speaker = captured["ontology"]["entity_types"][0]
    assert {item["name"] for item in speaker["attributes"]} == {"role", "entity_summary"}


def test_graph_builder_stores_an_ontology_graphiti_can_use():
    captured = {}

    class GraphApi:
        def set_ontology(self, **kwargs):
            captured.update(kwargs)

    class Client:
        graph = GraphApi()

    builder = object.__new__(GraphBuilderService)
    builder.client = Client()
    builder.set_ontology("graph-id", {
        "entity_types": [{
            "name": "Speaker",
            "attributes": ["graph_id", "labels"] + [
                f"field_{index}" for index in range(10)
            ],
        }],
        "edge_types": [{
            "name": "MENTIONS",
            "attributes": [],
            "source_targets": [{"source": "Speaker", "target": "Speaker"}],
        }],
    })

    assert captured["graph_ids"] == ["graph-id"]

    ontology = captured["ontology"]
    speaker = ontology["entity_types"][0]
    assert len(speaker["attributes"]) == MAX_ONTOLOGY_ATTRIBUTES
    assert speaker["attributes"][0]["name"] == "entity_graph_id"
    assert speaker["attributes"][1]["name"] == "entity_labels"

    models = build_ontology_models(ontology)
    assert "entity_graph_id" in models["entity_types"]["Speaker"].model_fields
    assert "details" in models["edge_types"]["MENTIONS"].model_fields
    assert models["edge_type_map"] == {("Speaker", "Speaker"): ["MENTIONS"]}


def test_graphiti_accepts_every_reserved_attribute_after_renaming():
    from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types

    captured = {}

    class GraphApi:
        def set_ontology(self, **kwargs):
            captured.update(kwargs)

    class Client:
        graph = GraphApi()

    builder = object.__new__(GraphBuilderService)
    builder.client = Client()
    builder.set_ontology("graph-id", {
        "entity_types": [{
            "name": "Speaker",
            "attributes": ["uuid", "name", "summary", "labels", "attributes", "created_at", "group_id"],
        }],
        "edge_types": [],
    })

    assert validate_entity_types(build_ontology_models(captured["ontology"])["entity_types"])


def test_graph_builder_supports_an_edge_only_ontology():
    captured = {}

    class GraphApi:
        def set_ontology(self, **kwargs):
            captured.update(kwargs)

    class Client:
        graph = GraphApi()

    builder = object.__new__(GraphBuilderService)
    builder.client = Client()
    builder.set_ontology("graph-id", {
        "entity_types": [],
        "edge_types": [{
            "name": "RELATED_TO",
            "attributes": ["reason"],
            "source_targets": [{"source": "Entity", "target": "Entity"}],
        }],
    })

    assert captured["ontology"]["entity_types"] == []
    models = build_ontology_models(captured["ontology"])
    assert models["entity_types"] is None
    assert list(models["edge_types"]) == ["RELATED_TO"]


def test_graph_builder_deduplicates_and_caps_edge_source_targets():
    captured = {}

    class GraphApi:
        def set_ontology(self, **kwargs):
            captured.update(kwargs)

    class Client:
        graph = GraphApi()

    source_targets = [
        {"source": f"Source{index}", "target": f"Target{index}"}
        for index in range(MAX_ONTOLOGY_SOURCE_TARGETS + 2)
    ]
    source_targets.insert(1, dict(source_targets[0]))

    builder = object.__new__(GraphBuilderService)
    builder.client = Client()
    builder.set_ontology("graph-id", {
        "entity_types": [],
        "edge_types": [{
            "name": "RELATED_TO",
            "attributes": ["reason"],
            "source_targets": source_targets,
        }],
    })

    normalized_targets = captured["ontology"]["edge_types"][0]["source_targets"]
    assert len(normalized_targets) == MAX_ONTOLOGY_SOURCE_TARGETS
    assert [(item["source"], item["target"]) for item in normalized_targets] == [
        (f"Source{index}", f"Target{index}")
        for index in range(MAX_ONTOLOGY_SOURCE_TARGETS)
    ]


def test_generator_ignores_invalid_entries_and_normalizes_edge_names():
    source_targets = [
        {"source": "speaker", "target": "news outlet"},
        {"source": "speaker", "target": "news outlet"},
        None,
    ] + [
        {"source": "speaker", "target": "news outlet" if index == 0 else "Person"}
        for index in range(12)
    ]

    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": ["speaker", None, 7, {"name": "news outlet"}],
        "edge_types": [
            "unusable edge",
            None,
            {"name": "worksFor", "source_targets": source_targets},
            {"name": "works-for", "source_targets": []},
        ],
    })

    assert [entity["name"] for entity in result["entity_types"][:2]] == [
        "Speaker",
        "NewsOutlet",
    ]
    assert [edge["name"] for edge in result["edge_types"]] == ["WORKS_FOR"]
    assert result["edge_types"][0]["source_targets"] == [
        {"source": "Speaker", "target": "NewsOutlet"},
        {"source": "Speaker", "target": "Person"},
    ]


def test_generator_caps_after_discarding_invalid_edge_endpoints():
    invalid_first = [
        {"source": f"Removed{index}", "target": "AlsoRemoved"}
        for index in range(MAX_ONTOLOGY_SOURCE_TARGETS)
    ]
    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "person"}, {"name": "organization"}],
        "edge_types": [{
            "name": "works_for",
            "source_targets": invalid_first + [
                {"source": "person", "target": "organization"}
            ],
        }],
    })

    assert result["edge_types"][0]["source_targets"] == [
        {"source": "Person", "target": "Organization"}
    ]
