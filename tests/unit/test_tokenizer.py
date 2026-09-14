from pathlib import Path

import yaml

from hai.data.pipeline import sha256_file, write_jsonl
from hai.tokenization.bpe import train_tokenizer, verify_tokenizer


def test_bpe_training_uses_train_only_and_reloads(tmp_path: Path) -> None:
    train_path = tmp_path / "data/processed/train.jsonl"
    write_jsonl(
        (
            {"id": f"r-{index}", "text": f"A small story about number {index}."}
            for index in range(30)
        ),
        train_path,
    )
    manifest_path = tmp_path / "data/manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "fixture",
                "processed": {
                    "train": {
                        "path": "data/processed/train.jsonl",
                        "sha256": sha256_file(train_path),
                        "records": 30,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "tokenizer.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "id": "fixture-bpe",
                "type": "bpe",
                "vocab_size": 64,
                "train_split_only": True,
                "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
                "output_dir": "artifacts/tokenizers/fixture-bpe",
            }
        ),
        encoding="utf-8",
    )
    metadata = train_tokenizer(config_path, manifest_path, tmp_path)
    assert metadata["pretrained_vocabulary"] is False
    assert verify_tokenizer(tmp_path / "artifacts/tokenizers/fixture-bpe", tmp_path)["ok"]
