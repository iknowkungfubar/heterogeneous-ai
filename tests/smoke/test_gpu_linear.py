import pytest


@pytest.mark.gpu
def test_gpu_linear_learning() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("compatible GPU runtime not available")
    from hai.training.linear_smoke import train

    weight, bias, _ = train(300)
    assert abs(weight - 3) < 0.25
    assert abs(bias - 2) < 0.25
