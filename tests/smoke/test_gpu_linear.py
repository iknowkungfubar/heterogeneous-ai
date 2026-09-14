import pytest


@pytest.mark.gpu
def test_gpu_linear_learning() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("compatible GPU runtime not available")
    from hai.training.linear_smoke import train

    weight, bias, _ = train(2000)
    assert abs(weight - 3) < 0.25
    assert abs(bias - 2) < 0.25


@pytest.mark.gpu
def test_gpu_fundamentals_and_checkpoint() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("compatible GPU runtime not available")
    from torch import nn

    from hai.training.linear_smoke import checkpoint_round_trip

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.tensor([[1.0], [2.0]], device=device)
    targets = torch.tensor([[3.0], [6.0]], device=device)
    initial = nn.functional.mse_loss(model(inputs), targets)
    initial.backward()
    assert model.weight.grad is not None
    assert torch.isfinite(initial)
    before = model.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.weight.detach())
    assert torch.cuda.max_memory_allocated(device) > 0
    assert checkpoint_round_trip(str(device))
