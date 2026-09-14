from pathlib import Path

from hai.memory.store import MemoryStore, self_test


def test_memory_self_test_passes() -> None:
    assert self_test()["ok"] is True


def test_memory_round_trip_and_type_query(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    try:
        item = store.add(
            "episodic",
            "a recorded event",
            source_id="event-1",
            source_kind="human_note",
            retention_until="2099-01-01T00:00:00+00:00",
            metadata={"session": "p16"},
        )
        rows = store.inspect("episodic")

        assert rows[0]["id"] == item["id"]
        assert rows[0]["metadata"] == {"session": "p16"}
        assert store.inspect("semantic") == []
    finally:
        store.close()
