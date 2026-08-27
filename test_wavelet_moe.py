"""Smoke test for the fixed-Haar Wavelet Frequency Expert used by WDR-MoE."""

import torch

from ultralytics.nn.modules.moe.experts import WaveletFrequencyExpert


def main() -> None:
    x = torch.randn(2, 32, 65, 67, requires_grad=True)
    outputs = []

    for band in ("ll", "lh", "hl", "hh"):
        expert = WaveletFrequencyExpert(
            in_channels=32,
            out_channels=32,
            expand_ratio=2,
            frequency_band=band,
        )
        y = expert(x)
        assert y.shape == x.shape, (band, x.shape, y.shape)
        assert torch.isfinite(y).all(), band
        outputs.append(y)

    torch.stack(outputs).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("PASS: fixed-Haar WFE output and gradient checks")


if __name__ == "__main__":
    main()
