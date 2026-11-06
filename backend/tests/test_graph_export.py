import json
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from flask import Flask

from app.api import graph as graph_api
from app.api import graph_bp
from app.config import Config
from app.utils.graph_client import NotFoundError

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}

GRAPH = {
    "graph_id": "graph-1",
    "nodes": [
        {
            "uuid": "n1",
            "name": "Ada & <Co>",
            "labels": ["Person"],
            "summary": "bad\x0bchar hello",
            "attributes": {"role": "operator"},
            "created_at": "2026-01-01",
        },
        {
            "uuid": "n2",
            "name": "GS1",
            "labels": [],
            "summary": "",
            "attributes": {},
            "created_at": None,
        },
    ],
    "edges": [
        {
            "uuid": "e1",
            "name": "WORKS_WITH",
            "fact": "Ada works with GS1",
            "fact_type": "WORKS_WITH",
            "source_node_uuid": "n1",
            "target_node_uuid": "n2",
            "attributes": {},
            "created_at": "2026-01-02",
            "valid_at": None,
            "invalid_at": None,
            "expired_at": None,
            "episodes": ["ep1"],
        }
    ],
    "node_count": 2,
    "edge_count": 1,
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_STORE_URL", "test")
    monkeypatch.setattr(
        graph_api,
        "GraphBuilderService",
        lambda api_key: SimpleNamespace(get_graph_data=lambda graph_id: GRAPH),
    )
    app = Flask(__name__)
    app.register_blueprint(graph_bp, url_prefix="/api/graph")
    return app.test_client()


def test_json_export_downloads_all_graph_data(client):
    response = client.get("/api/graph/graph-1/export")

    assert response.status_code == 200
    assert response.mimetype == "application/json"
    assert 'filename="graph_graph-1.json"' in response.headers["Content-Disposition"]
    assert json.loads(response.data) == GRAPH


def test_graphml_export_is_valid_and_complete(client):
    response = client.get("/api/graph/graph-1/export?format=graphml")

    assert response.status_code == 200
    assert 'filename="graph_graph-1.graphml"' in response.headers["Content-Disposition"]

    root = ElementTree.fromstring(response.data)
    nodes = root.findall(".//g:node", NS)
    edges = root.findall(".//g:edge", NS)
    assert [node.get("id") for node in nodes] == ["n1", "n2"]
    assert (edges[0].get("source"), edges[0].get("target")) == ("n1", "n2")

    values = {d.get("key"): d.text for d in nodes[0].findall("g:data", NS)}
    assert values["n_name"] == "Ada & <Co>"
    assert values["n_summary"] == "badchar hello"
    assert json.loads(values["n_labels"]) == ["Person"]

    edge_values = {d.get("key"): d.text for d in edges[0].findall("g:data", NS)}
    assert edge_values["e_fact"] == "Ada works with GS1"


def test_export_rejects_bad_requests(client):
    assert client.get("/api/graph/graph-1/export?format=xml").status_code == 400
    assert client.get("/api/graph/bad.id/export").status_code == 400


def test_missing_graph_returns_404(client, monkeypatch):
    def missing(graph_id):
        raise NotFoundError(body={})

    monkeypatch.setattr(
        graph_api,
        "GraphBuilderService",
        lambda api_key: SimpleNamespace(get_graph_data=missing),
    )

    assert client.get("/api/graph/graph-1/export").status_code == 404


def test_zep_failure_is_not_exported_as_a_partial_file(client, monkeypatch):
    def broken(graph_id):
        raise RuntimeError("graph store unavailable")

    monkeypatch.setattr(
        graph_api,
        "GraphBuilderService",
        lambda api_key: SimpleNamespace(get_graph_data=broken),
    )

    response = client.get("/api/graph/graph-1/export")

    assert response.status_code == 500
    assert response.get_json()["success"] is False
