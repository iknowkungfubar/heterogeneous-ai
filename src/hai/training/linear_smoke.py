from __future__ import annotations

import argparse
import math


def train(steps: int = 500) -> tuple[float, float, float]:
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable: stop at P01/P02 rather than continuing model work")
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
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return float(model.weight.detach().cpu().item()), float(model.bias.detach().cpu().item()), final_loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    weight, bias, loss = train(args.steps)
    print(f"weight={weight:.6f} bias={bias:.6f} loss={loss:.8f}")
    if abs(weight - 3) > 0.15 or abs(bias - 2) > 0.15:
        raise SystemExit("smoke model did not converge sufficiently")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
