"""Record non-sensitive environment facts and verify real GPU computation."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import torch


def main():
    artifact = Path("artifacts/environment.json")
    artifact.parent.mkdir(exist_ok=True)
    evidence = json.loads(artifact.read_text()) if artifact.exists() else {}
    evidence.update(
        recorded_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        os=platform.system(),
        os_version=platform.mac_ver()[0],
        architecture=platform.machine(),
        packages={name: version(name) for name in (
            "torch", "numpy", "scipy", "scikit-learn", "matplotlib",
            "pytest", "reportlab", "pypdf",
        )},
        device={
            "cuda_available": torch.cuda.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
    )
    if not torch.backends.mps.is_available():
        evidence["mps_validation"] = {"status": "unavailable"}
        artifact.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2))
        raise SystemExit(2)

    torch.manual_seed(20260916)
    x = torch.randn(128, 64, dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, 32, dtype=torch.float32, requires_grad=True)
    y_cpu = x @ w
    loss_cpu = y_cpu.square().mean()
    loss_cpu.backward()
    x_gpu = x.detach().to("mps").requires_grad_()
    w_gpu = w.detach().to("mps").requires_grad_()
    torch.mps.synchronize()
    start = time.perf_counter()
    y_gpu = x_gpu @ w_gpu
    loss_gpu = y_gpu.square().mean()
    loss_gpu.backward()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    for actual, expected in (
        (y_gpu.cpu(), y_cpu.detach()),
        (x_gpu.grad.cpu(), x.grad),
        (w_gpu.grad.cpu(), w.grad),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    evidence["mps_validation"] = {
        "status": "passed",
        "operation": "float32 matrix multiplication, squared mean loss, backward",
        "input_shapes": [[128, 64], [64, 32]],
        "output_device": str(y_gpu.device),
        "gradient_device": str(w_gpu.grad.device),
        "loss": loss_gpu.item(),
        "cpu_loss": loss_cpu.item(),
        "forward_max_absolute_error": (y_gpu.detach().cpu() - y_cpu.detach()).abs().max().item(),
        "x_gradient_max_absolute_error": (x_gpu.grad.cpu() - x.grad).abs().max().item(),
        "w_gradient_max_absolute_error": (w_gpu.grad.cpu() - w.grad).abs().max().item(),
        "comparison_atol": 2e-5,
        "comparison_rtol": 1e-4,
        "synchronized_forward_backward_seconds": elapsed,
        "timing_note": "Single verification including first-use overhead; not a benchmark.",
    }
    artifact.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
