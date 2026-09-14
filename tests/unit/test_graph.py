from hai.graph.store import GraphStore, self_test


def test_graph_self_test_passes():
    result = self_test()

    assert result["ok"] is True
    assert all(result["checks"].values())


def test_graph_filters_untrusted_edges_and_preserves_provenance():
    store = GraphStore(":memory:")
    try:
        source = store.add_entity(
            "Source", "person", source_id="note-1", source_kind="human_note"
        )
        target = store.add_entity(
            "Target", "person", source_id="note-1", source_kind="human_note"
        )
        store.validate_entity(source["id"], "entity-source")
        store.validate_entity(target["id"], "entity-target")
        verified = store.add_relation(
            source["id"],
            "knows",
            target["id"],
            source_id="note-1",
            source_kind="human_note",
        )
        untrusted = store.add_relation(
            source["id"],
            "guesses",
            target["id"],
            source_id="model-1",
            source_kind="model_output",
        )
        store.validate_relation(verified["id"], "relation-evidence")

        assert [item["relation_type"] for item in store.neighbors(source["id"])] == ["knows"]
        assert len(store.neighbors(source["id"], min_trust="untrusted")) == 2
        assert store.get_relation(verified["id"])["validation_evidence_id"] == "relation-evidence"
        assert untrusted["trust_level"] == "untrusted"
    finally:
        store.close()


def test_conflicts_are_retained_and_paths_are_deterministic():
    store = GraphStore(":memory:")
    try:
        first = store.add_entity("A", "concept", source_id="fixture", source_kind="fixture")
        second = store.add_entity("B", "concept", source_id="fixture", source_kind="fixture")
        third = store.add_entity("C", "concept", source_id="fixture", source_kind="fixture")
        for entity in (first, second, third):
            store.validate_entity(entity["id"], f"evidence-{entity['label']}")
        direct = store.add_relation(
            first["id"], "maps_to", second["id"], source_id="fixture", source_kind="fixture",
            conflict_key="a-mapping",
        )
        alternate = store.add_relation(
            first["id"], "maps_to", third["id"], source_id="model", source_kind="model_output",
            conflict_key="a-mapping",
        )
        store.validate_relation(direct["id"], "mapping-evidence")

        assert alternate["status"] == "conflict"
        assert store.path(first["id"], second["id"], max_hops=1)["hops"] == 1
        assert store.path(first["id"], third["id"], max_hops=1) is None
        assert store.add_relation(
            first["id"], "maps_to", second["id"], source_id="fixture", source_kind="fixture",
            conflict_key="a-mapping",
        )["status"] == "duplicate"
    finally:
        store.close()
