from types import SimpleNamespace

from app.services.graph_entity_reader import GraphEntityReader




def test_get_node_edges_uses_the_supported_sdk_method():
    calls = []

    class NodeApi:
        def get_edges(self, *, node_uuid):
            calls.append(node_uuid)
            return [SimpleNamespace(
                uuid_="edge-1",
                name="KNOWS",
                fact="Alice knows Bob",
                source_node_uuid="node-1",
                target_node_uuid="node-2",
                attributes={"since": "2024"},
            )]

    class GraphApi:
        node = NodeApi()

    class Client:
        graph = GraphApi()

    reader = object.__new__(GraphEntityReader)
    reader.client = Client()

    assert reader.get_node_edges("node-1") == [{
        "uuid": "edge-1",
        "name": "KNOWS",
        "fact": "Alice knows Bob",
        "source_node_uuid": "node-1",
        "target_node_uuid": "node-2",
        "attributes": {"since": "2024"},
    }]
    assert calls == ["node-1"]
