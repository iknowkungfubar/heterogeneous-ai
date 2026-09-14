from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path


def train(steps: int = 2000, seed: int = 1337) -> tuple[float, float, float]:
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable: stop at P01/P02 rather than continuing model work")
    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = torch.linspace(-10, 10, 10_000, device=device).unsqueeze(1)
    y = 3 * x + 2
    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    final_loss = math.inf
    for _ in range(steps):
        prediction = model(x)
        loss = loss_fn(prediction, y)
        if not torch.isfinite(loss):
            raise RuntimeError("linear smoke loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return (
        float(model.weight.detach().cpu().item()),
        float(model.bias.detach().cpu().item()),
        final_loss,
    )


def checkpoint_round_trip(device: str = "cuda") -> bool:
    import torch
    from torch import nn

    model = nn.Linear(1, 1).to(device)
    with tempfile.TemporaryDirectory(prefix="hai-checkpoint-") as directory:
        path = Path(directory) / "linear.pt"
        torch.save(model.state_dict(), path)
        restored = nn.Linear(1, 1).to(device)
        state = torch.load(path, map_location=device, weights_only=True)
        restored.load_state_dict(state)
    return all(
        torch.equal(left, right)
        for left, right in zip(model.parameters(), restored.parameters(), strict=True)
    )


def main() -> int:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    torch.cuda.reset_peak_memory_stats()
    weight, bias, loss = train(args.steps)
    peak_memory = torch.cuda.max_memory_allocated()
    checkpoint_ok = checkpoint_round_trip()
    print(
        f"weight={weight:.6f} bias={bias:.6f} loss={loss:.8f} "
        f"peak_memory_bytes={peak_memory} checkpoint_round_trip={checkpoint_ok}"
    )
    if abs(weight - 3) > 0.15 or abs(bias - 2) > 0.15:
        raise SystemExit("smoke model did not converge sufficiently")
    if not checkpoint_ok:
        raise SystemExit("checkpoint round-trip failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
