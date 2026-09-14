# Security and Safety Model

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's private vulnerability reporting for this repository when available, or contact the maintainer privately through the GitHub profile. Include a concise impact summary, affected commit or path, reproduction details that do not contain secrets, and any suggested mitigation. Do not include credentials, private data, or live exploit payloads in a public pull request.

## Threats in scope

- malicious/poisoned datasets;
- archive path traversal and unexpected file types;
- prompt injection through retrieved/memory content;
- arbitrary shell/tool invocation;
- poisoned memory becoming trusted knowledge;
- checkpoint tampering;
- dependency compromise;
- secret leakage;
- unsafe network exposure;
- model/component denial of service or malformed output.

## Principles

1. Neural models do not receive unrestricted host shell access.
2. Tools are exposed through an allowlisted structured operation layer.
3. Untrusted retrieved/model text does not automatically become trusted memory.
4. Provenance and trust state travel with evidence/memory.
5. Canonical artifacts are checksummed before promotion/loading.
6. Local serving binds to loopback by default.
7. Secrets never enter prompts, datasets, logs, or Git unless deliberately redacted and safe.
8. RL actions are restricted to a finite registry of safe orchestration operations.

## Memory ingestion

`candidate → source/provenance check → duplicate/conflict check → validation/trust label → memory`

Unverified content remains explicitly unverified.

## Incident response

If secret/private data is committed or logged: stop publication/sync, preserve enough evidence to understand scope, rotate affected credentials, purge using appropriate repository-history procedures, and document the incident without reproducing the secret.
