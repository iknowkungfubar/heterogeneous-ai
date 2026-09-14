from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import yaml

from hai.data.pipeline import _repo_path, sha256_file


class RetrievalError(ValueError):
    """Raised when retrieval artifacts or split boundaries are invalid."""


def load_retrieval_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("id") != "dual-encoder-v1":
        raise RetrievalError("retrieval config must declare dual-encoder-v1")
    return config


def _manifest(root: Path) -> tuple[dict, Path]:
    path = root / "data" / "manifests" / "tinystories-smoke.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    return manifest, path


def _records(root: Path, split: str, limit: int | None = None) -> tuple[list[dict], str]:
    manifest, _ = _manifest(root)
    details = manifest["processed"][split]
    path = _repo_path(root, details["path"], f"retrieval {split}")
    if sha256_file(path) != details["sha256"]:
        raise RetrievalError(f"retrieval {split} checksum verification failed")
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records, details["sha256"]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _query(record: dict, query_tokens: int) -> str:
    return " ".join(_tokens(record["text"])[:query_tokens])


def _vocabulary(records: list[dict], size: int) -> dict[str, int]:
    counts = Counter(token for record in records for token in _tokens(record["text"]))
    words = [word for word, _ in counts.most_common(max(1, size - 1))]
    return {"<unk>": 0, **{word: index + 1 for index, word in enumerate(words)}}


def _ids(text: str, vocabulary: dict[str, int], max_tokens: int) -> list[int]:
    values = [vocabulary.get(token, 0) for token in _tokens(text)[:max_tokens]]
    return values or [0]


def _model_and_torch(checkpoint: dict):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RetrievalError("PyTorch is required for scratch retrieval training") from exc

    class DualEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                checkpoint["vocab_size"], checkpoint["embedding_dim"], padding_idx=0
            )

        def forward(self, token_batches):
            vectors = []
            for tokens in token_batches:
                embedded = self.embedding(tokens)
                vectors.append(embedded.mean(dim=1))
            return vectors

    return DualEncoder(), torch


def train_encoder(config_path: Path, root: Path) -> dict:
    config = load_retrieval_config(config_path)
    records, train_hash = _records(root, "train", config["train_records"])
    checkpoint = {
        "config": config,
        "vocab": _vocabulary(records, config["vocab_size"]),
        "vocab_size": min(config["vocab_size"], len(_vocabulary(records, config["vocab_size"]))),
        "embedding_dim": config["embedding_dim"],
        "train_hash": train_hash,
        "train_records": len(records),
    }
    model, torch = _model_and_torch(checkpoint)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    losses = []
    model.train()
    for step in range(config["training_steps"]):
        start = (step * config["batch_size"]) % len(records)
        batch = [records[(start + offset) % len(records)] for offset in range(config["batch_size"])]
        queries = _tensor_batch(
            [_query(record, config["query_tokens"]) for record in batch], checkpoint, config, torch
        )
        documents = _tensor_batch(
            [record["text"] for record in batch], checkpoint, config, torch
        )
        query_vectors, document_vectors = model([queries, documents])
        scores = query_vectors @ document_vectors.T / config["temperature"]
        labels = torch.arange(len(batch))
        loss = torch.nn.functional.cross_entropy(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint["state_dict"] = model.state_dict()
    torch.save(checkpoint, output / "model.pt")
    artifact_hash = sha256_file(output / "model.pt")
    metadata = {
        "retrieval_id": config["id"],
        "initialization": config["initialization"],
        "seed": config["seed"],
        "train_split": "train",
        "train_hash": train_hash,
        "train_records": len(records),
        "embedding_dim": config["embedding_dim"],
        "training_steps": config["training_steps"],
        "start_loss": losses[0],
        "final_loss": losses[-1],
        "model_sha256": artifact_hash,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _load_checkpoint(config: dict, root: Path):
    try:
        import torch
    except ImportError as exc:
        raise RetrievalError("PyTorch is required for retrieval") from exc
    path = root / config["output_dir"] / "model.pt"
    if not path.is_file():
        raise RetrievalError(f"retrieval checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, _ = _model_and_torch(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model, torch


def build_index(config_path: Path, root: Path, split: str | None = None) -> dict:
    config = load_retrieval_config(config_path)
    index_split = split or config["index_split"]
    if index_split == "test":
        raise RetrievalError("TEST indexing is reserved for final evaluation after freeze")
    checkpoint, model, torch = _load_checkpoint(config, root)
    records, split_hash = _records(root, index_split)
    vectors = _dense_vectors(records, checkpoint, model, torch, root, config)
    try:
        import hnswlib
    except ImportError as exc:
        raise RetrievalError("hnswlib is required to build the dense retrieval index") from exc
    index = hnswlib.Index(space="cosine", dim=config["embedding_dim"])
    index.init_index(max_elements=len(records), ef_construction=100, M=16)
    index.add_items(vectors, list(range(len(records))))
    index.set_ef(50)
    output = root / config["output_dir"]
    index.save_index(str(output / "index.bin"))
    documents = output / "documents.jsonl"
    documents.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    metadata = {
        "index_split": index_split,
        "index_hash": split_hash,
        "documents": len(records),
        "model_sha256": sha256_file(output / "model.pt"),
        "index_sha256": sha256_file(output / "index.bin"),
    }
    (output / "index-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _dense_vectors(records, checkpoint, model, torch, root, config):
    batches = _tensor_batch([record["text"] for record in records], checkpoint, config, torch)
    with torch.no_grad():
        return model([batches, batches])[1].numpy()


def _tensor_batch(texts, checkpoint, config, torch):
    rows = [_ids(text, checkpoint["vocab"], config["max_tokens"]) for text in texts]
    width = max(len(row) for row in rows)
    return torch.tensor([row + [0] * (width - len(row)) for row in rows])


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _lexical(query: str, document: str) -> float:
    query_tokens, document_tokens = set(_tokens(query)), set(_tokens(document))
    return len(query_tokens & document_tokens) / len(query_tokens) if query_tokens else 0.0


def evaluate_retrieval(config_path: Path, root: Path, split: str = "validation") -> dict:
    config = load_retrieval_config(config_path)
    if split == "test":
        raise RetrievalError(
            "TEST retrieval evaluation is reserved for final evaluation after freeze"
        )
    checkpoint, model, torch = _load_checkpoint(config, root)
    index_metadata = json.loads(
        (root / config["output_dir"] / "index-metadata.json").read_text(encoding="utf-8")
    )
    if index_metadata["index_split"] != split:
        raise RetrievalError("retrieval index split must match evaluation split")
    records, split_hash = _records(root, split, config["evaluation_records"])
    stored_records, _ = _records(root, split)
    vectors = _dense_vectors(stored_records, checkpoint, model, torch, root, config)
    query_vectors = _dense_vectors(
        [{"text": _query(record, config["query_tokens"])} for record in records],
        checkpoint,
        model,
        torch,
        root,
        config,
    )
    try:
        import hnswlib
        import numpy as np
    except ImportError as exc:
        raise RetrievalError("hnswlib is required for dense retrieval evaluation") from exc
    index = hnswlib.Index(space="cosine", dim=config["embedding_dim"])
    index.load_index(
        str(root / config["output_dir"] / "index.bin"), max_elements=len(stored_records)
    )
    index.set_ef(50)
    document_token_sets = [set(_tokens(record["text"])) for record in stored_records]
    vector_norms = np.linalg.norm(vectors, axis=1)
    results = {}
    for method in ("random", "lexical", "dense", "hybrid"):
        rankings = []
        rng = random.Random(config["seed"])
        for query_index, record in enumerate(records):
            query = _query(record, config["query_tokens"])
            if method == "dense":
                query_vector = query_vectors[query_index]
                labels, distances = index.knn_query(query_vector, k=min(10, len(stored_records)))
                ranking = list(labels[0])
            else:
                query_tokens = set(_tokens(query))
                lexical_scores = [
                    len(query_tokens & document_tokens) / len(query_tokens)
                    if query_tokens
                    else 0.0
                    for document_tokens in document_token_sets
                ]
                dense_scores = vectors @ query_vectors[query_index]
                query_norm = np.linalg.norm(query_vectors[query_index])
                if query_norm:
                    dense_scores = dense_scores / (vector_norms * query_norm)
                scores = []
                for position, _candidate in enumerate(stored_records):
                    if method == "random":
                        score = rng.random()
                    elif method == "lexical":
                        score = lexical_scores[position]
                    else:
                        score = 0.5 * lexical_scores[position] + 0.5 * dense_scores[position]
                    scores.append((score, position))
                ranking = [position for _, position in sorted(scores, reverse=True)[:10]]
            rankings.append(ranking)
        results[method] = _ranking_metrics(
            rankings, records, stored_records, index_metadata["index_hash"], split
        )
    return {
        "split": split,
        "samples": len(records),
        "index_split": index_metadata["index_split"],
        "index_hash": index_metadata["index_hash"],
        "query_split_hash": split_hash,
        "methods": results,
        "provenance_fields": ["document_id", "split", "source_sha256", "rank", "score"],
    }


def _ranking_metrics(rankings, queries, documents, source_sha256: str, split: str) -> dict:
    ranks = []
    for ranking, query in zip(rankings, queries, strict=True):
        target = query["id"]
        try:
            rank = next(
                index + 1
                for index, position in enumerate(ranking)
                if documents[position]["id"] == target
            )
        except StopIteration:
            rank = None
        ranks.append(rank)
    return {
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks),
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / len(ranks),
        "mrr": sum(1 / rank if rank is not None else 0.0 for rank in ranks) / len(ranks),
        "provenance_sample": {
            "document_id": documents[rankings[0][0]]["id"],
            "split": split,
            "source_sha256": source_sha256,
            "rank": 1,
            "score": 0.0,
        },
    }
