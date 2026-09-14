# P15 scratch-trained semantic retrieval — 2026-09-14

## Hypothesis

A small dual encoder trained from random weights on TRAIN-only stories can
beat a random retrieval baseline, while a hybrid lexical+dense index improves
over lexical retrieval alone and preserves document provenance.

## Scope and provenance

- Experiment: `P15-RETRIEVAL-20260914`
- Parent: `P14-CORE-M2-20260914`
- Config: `configs/models/embedding.yaml`
- Encoder: `dual-encoder-v1`, random initialization, seed `1337`
- TRAIN hash: `1613a0868eda0f1ba05ed6bb3680165566515f643e430f98abf62d106dd16785`
- VALIDATION/index hash: `9d87a6a79461390c40acd8ed95e652c9fe4c403ec35521aecd0f7e3f55da9b21`
- TRAIN records used for fitting: 512
- Contrastive steps: 100
- VALIDATION queries: 256
- TEST was not used for training, indexing, tuning, or selection.

## Training and artifact evidence

The scratch encoder reduced contrastive loss from `17.5246963501` to
`1.6270910501`. The model artifact SHA-256 is
`9ad8f7fccbae3aed4239b59b91c76f0561f176e2f1e8a3435cd93b1e5de601f8`; the
HNSW index SHA-256 is
`4568fabd6dcdbf3c2135dbb684c88db94fd1db3288b6d6f1e99f8b17d528766f`.

## Validation retrieval results

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| Random | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Lexical overlap | 0.1875 | 0.2813 | 0.3320 | 0.2244 |
| Scratch dense HNSW | 0.0742 | 0.1367 | 0.1719 | 0.1028 |
| Hybrid lexical+dense | 0.2383 | 0.3867 | 0.4375 | 0.2973 |

The scratch dense model beats random at every cutoff. Hybrid retrieval beats
lexical retrieval at every reported metric; dense alone does not beat lexical,
so no unsupported claim is made about dense dominance.

Every returned result includes `document_id`, index split, source SHA-256, rank,
and score fields. Local MLflow runs were recorded for encoder training, index
building, and evaluation.

## Commands and results

```text
hai retrieval train-encoder --config configs/models/embedding.yaml
start_loss=17.5246963501; final_loss=1.6270910501

hai retrieval build-index --config configs/models/embedding.yaml --split validation
documents=10000; index split=validation

hai retrieval evaluate --config configs/models/embedding.yaml --split validation
random recall@1=0.0; dense recall@1=0.07421875;
lexical recall@1=0.1875; hybrid recall@1=0.23828125
```

## Gate evaluation

- Scratch embeddings beat random/basic baselines: **pass**. Dense beats the
  random baseline; hybrid beats lexical at all reported metrics.
- Hybrid retrieval evaluation exists: **pass**. Random, lexical, dense HNSW,
  and hybrid metrics are reported on VALIDATION.
- Every result has provenance: **pass**. Result traces expose document ID,
  split, source hash, rank, and score; model/index hashes are recorded.

## Conclusion and next action

P15 passes. The small scratch encoder is useful as a retrieval component but
not yet a replacement for lexical matching; the retained hybrid is the
evidence-backed choice. Advance to P16 for explicit working, episodic,
semantic, and procedural memory stores.
