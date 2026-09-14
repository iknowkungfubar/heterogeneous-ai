# Model Card: gru-sequence-v1

## Purpose

Independent recurrent sequence specialist for measuring whether structurally
different sequence errors complement the Transformer baseline.

## Architecture and exact parameter count

Embedding size 384, three GRU layers, hidden size 512, context length 512, and
an untied vocabulary projection. Exact parameter count: **11,879,424**.

## Initialization

All weights were randomly initialized. No pretrained model or vocabulary was
loaded.

## Training data and tokenizer

The model uses the same governed TinyStories train/validation/test manifest and
the same project-owned `bpe-8192-v1` tokenizer as TFRM-0001. TRAIN, VALIDATION,
and TEST roles remain separate.

## Training and evaluation

Experiment: `GRU-0001`; seed `1337`; CUDA; 100 steps.

- Training loss: `9.010451316833496` to `5.95500373840332`.
- VALIDATION loss: `6.0092668533325195`.
- VALIDATION perplexity: `407.1846923828125`.
- Checkpoint SHA-256:
  `1ff7c9166c171776eaa3370607d3316feea5ffb9e667f594205b8551fb8839eb`.

## Complementarity

Compared with the frozen Transformer on 64 identical VALIDATION blocks:

| Outcome | Correct token predictions |
| --- | ---: |
| Both correct | 439 |
| Transformer only | 4,828 |
| GRU only | 1,948 |
| Both wrong | 25,553 |

The token-level complementarity rate was `20.6787%`, defined as the fraction of
predictions where exactly one specialist was correct.

## Limitations

This is a short, smoke-scale comparison with unequal parameter counts and
training budgets. The high both-wrong count means complementarity is not evidence
of adequate language quality. A fairer comparison requires a preregistered
matched-budget experiment.
