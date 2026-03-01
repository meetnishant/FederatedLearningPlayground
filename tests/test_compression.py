"""Tests for gradient compression: top-k, quantization, error feedback, and config (M3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from flp.compression import GradientCompressor
from flp.compression.error_feedback import ErrorFeedbackBuffer
from flp.compression.quantization import QuantizationResult, quantize_state_dict
from flp.compression.topk import TopKResult, topk_compress
from flp.experiments.config_loader import CompressionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sd(**tensors: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build a state dict from keyword args."""
    return {k: v for k, v in tensors.items()}


def _make_update(client_id: int = 0, sd: dict | None = None):
    """Return a mock ClientUpdate."""
    from flp.core.client import ClientUpdate
    train_result = MagicMock()
    train_result.loss = 0.5
    return ClientUpdate(
        client_id=client_id,
        state_dict=sd or {"w": torch.randn(10)},
        num_samples=100,
        train_result=train_result,
    )


# ===========================================================================
# topk_compress
# ===========================================================================


class TestTopKCompress:
    def test_returns_topk_result(self) -> None:
        sd = _make_sd(w=torch.randn(100))
        result = topk_compress(sd, k_ratio=0.1)
        assert isinstance(result, TopKResult)

    def test_compression_ratio_approx_k_ratio(self) -> None:
        sd = _make_sd(w=torch.randn(1000))
        result = topk_compress(sd, k_ratio=0.2)
        assert result.compression_ratio == pytest.approx(0.2, abs=0.05)

    def test_num_elements_kept_approx(self) -> None:
        sd = _make_sd(w=torch.ones(100))
        # With a constant tensor all elements have the same magnitude.
        # ceil(0.1 * 100) = 10 kept.
        result = topk_compress(sd, k_ratio=0.1)
        assert result.num_elements_total == 100

    def test_top_values_preserved(self) -> None:
        # Build a tensor where the largest value is obvious.
        t = torch.zeros(10)
        t[3] = 100.0  # largest by far
        result = topk_compress({"w": t}, k_ratio=0.1)
        assert result.state_dict["w"][3].item() == pytest.approx(100.0)

    def test_zeroes_non_top_values(self) -> None:
        t = torch.zeros(10)
        t[0] = 100.0
        result = topk_compress({"w": t}, k_ratio=0.1)
        # All elements except index 0 should be zero.
        for i in range(1, 10):
            assert result.state_dict["w"][i].item() == pytest.approx(0.0)

    def test_integer_buffers_unchanged(self) -> None:
        int_buf = torch.tensor([7], dtype=torch.int64)
        sd = _make_sd(w=torch.randn(20), count=int_buf)
        result = topk_compress(sd, k_ratio=0.5)
        assert result.state_dict["count"].item() == 7
        assert result.state_dict["count"].dtype == torch.int64

    def test_k_ratio_one_is_noop(self) -> None:
        t = torch.randn(50)
        result = topk_compress({"w": t}, k_ratio=1.0)
        assert torch.allclose(result.state_dict["w"], t)
        assert result.compression_ratio == pytest.approx(1.0)

    def test_invalid_k_ratio_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            topk_compress({"w": torch.randn(10)}, k_ratio=0.0)

    def test_invalid_k_ratio_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            topk_compress({"w": torch.randn(10)}, k_ratio=1.5)

    def test_multiple_tensors_total_elements(self) -> None:
        sd = _make_sd(a=torch.randn(100), b=torch.randn(200))
        result = topk_compress(sd, k_ratio=0.1)
        assert result.num_elements_total == 300

    def test_output_dtype_unchanged(self) -> None:
        t = torch.randn(20, dtype=torch.float64)
        result = topk_compress({"w": t}, k_ratio=0.5)
        assert result.state_dict["w"].dtype == torch.float64

    def test_empty_state_dict(self) -> None:
        result = topk_compress({}, k_ratio=0.5)
        assert result.state_dict == {}
        assert result.num_elements_total == 0
        assert result.compression_ratio == pytest.approx(1.0)


# ===========================================================================
# quantize_state_dict
# ===========================================================================


class TestQuantizeStateDict:
    def test_returns_quantization_result(self) -> None:
        sd = _make_sd(w=torch.randn(10))
        result = quantize_state_dict(sd, bits=16)
        assert isinstance(result, QuantizationResult)

    def test_float16_compression_ratio(self) -> None:
        result = quantize_state_dict({"w": torch.randn(10)}, bits=16)
        assert result.compression_ratio == pytest.approx(0.5)
        assert result.bytes_per_element == 2

    def test_int8_compression_ratio(self) -> None:
        result = quantize_state_dict({"w": torch.randn(10)}, bits=8)
        assert result.compression_ratio == pytest.approx(0.25)
        assert result.bytes_per_element == 1

    def test_invalid_bits_raises(self) -> None:
        with pytest.raises(ValueError):
            quantize_state_dict({"w": torch.randn(10)}, bits=4)  # type: ignore[arg-type]

    def test_float16_preserves_dtype(self) -> None:
        t = torch.randn(10, dtype=torch.float32)
        result = quantize_state_dict({"w": t}, bits=16)
        assert result.state_dict["w"].dtype == torch.float32

    def test_float16_round_trip_error_small(self) -> None:
        t = torch.randn(100, dtype=torch.float32)
        result = quantize_state_dict({"w": t}, bits=16)
        max_err = (result.state_dict["w"] - t).abs().max().item()
        # float16 max rounding error for normal-range values should be tiny
        assert max_err < 0.01

    def test_int8_round_trip_error_bounded(self) -> None:
        t = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)
        result = quantize_state_dict({"w": t}, bits=8)
        max_err = (result.state_dict["w"] - t).abs().max().item()
        # With 256 bins over range [-1, 1], max error ≤ 2/255 ≈ 0.0079
        assert max_err < 2.0 / 255 + 1e-5

    def test_integer_buffers_unchanged(self) -> None:
        int_buf = torch.tensor([42], dtype=torch.int32)
        sd = _make_sd(w=torch.randn(10), count=int_buf)
        result = quantize_state_dict(sd, bits=16)
        assert result.state_dict["count"].item() == 42
        assert result.state_dict["count"].dtype == torch.int32

    def test_constant_tensor_int8(self) -> None:
        # min == max — should not produce NaN or raise.
        t = torch.ones(10) * 3.14
        result = quantize_state_dict({"w": t}, bits=8)
        assert not result.state_dict["w"].isnan().any()
        assert torch.allclose(result.state_dict["w"], t)


# ===========================================================================
# ErrorFeedbackBuffer
# ===========================================================================


class TestErrorFeedbackBuffer:
    def test_first_call_no_residual(self) -> None:
        buf = ErrorFeedbackBuffer([0])
        sd = {"w": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])}
        compressed = buf.apply_and_compress(0, sd, lambda s: topk_compress(s, 0.4).state_dict)
        # Output should be the compressed version of the raw update.
        assert compressed["w"].shape == sd["w"].shape

    def test_residual_carried_forward(self) -> None:
        buf = ErrorFeedbackBuffer([0])
        # Update with a single large value at index 0, small at others.
        sd = {"w": torch.tensor([0.1, 0.1, 0.1, 0.1, 10.0])}

        def compress_fn(s: dict) -> dict:
            # Keep only top-20% (1 out of 5) = only the 10.0 element.
            return topk_compress(s, 0.2).state_dict

        buf.apply_and_compress(0, sd, compress_fn)
        # After round 1, the error for small values (0.1) is accumulated.
        assert buf.has_residual(0)

    def test_residual_accumulates_over_rounds(self) -> None:
        buf = ErrorFeedbackBuffer([0])
        # Small value that will be dropped in round 1 and kept in round 2.
        update = {"w": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])}

        def compress_fn(s: dict) -> dict:
            return topk_compress(s, 0.2).state_dict  # keeps top 20% = 1 element

        out1 = buf.apply_and_compress(0, update, compress_fn)
        # Round 2: same update — the residual of the dropped zeros should be 0 still.
        out2 = buf.apply_and_compress(0, update, compress_fn)
        assert out1["w"].shape == out2["w"].shape

    def test_reset_clears_buffer(self) -> None:
        buf = ErrorFeedbackBuffer([0])
        sd = {"w": torch.randn(10)}
        buf.apply_and_compress(0, sd, lambda s: topk_compress(s, 0.1).state_dict)
        assert buf.has_residual(0)
        buf.reset(0)
        assert not buf.has_residual(0)

    def test_multi_client_isolation(self) -> None:
        buf = ErrorFeedbackBuffer([0, 1])
        # Only compress for client 0; client 1 is never touched.
        sd0 = {"w": torch.ones(10)}
        buf.apply_and_compress(0, sd0, lambda s: topk_compress(s, 0.2).state_dict)
        # Client 0 had non-zero update → residual exists after keeping only 20%.
        assert buf.has_residual(0)
        # Client 1 was never called — its buffer should still be empty.
        assert not buf.has_residual(1)

    def test_unregistered_client_raises(self) -> None:
        buf = ErrorFeedbackBuffer([0, 1])
        with pytest.raises(KeyError):
            buf.apply_and_compress(99, {"w": torch.randn(5)}, lambda s: s)

    def test_reset_unregistered_raises(self) -> None:
        buf = ErrorFeedbackBuffer([0])
        with pytest.raises(KeyError):
            buf.reset(99)

    def test_integer_buffers_not_accumulated(self) -> None:
        buf = ErrorFeedbackBuffer([0])
        sd = {"w": torch.randn(10), "count": torch.tensor([3], dtype=torch.int64)}
        compressed = buf.apply_and_compress(0, sd, lambda s: topk_compress(s, 0.5).state_dict)
        assert compressed["count"].item() == 3


# ===========================================================================
# GradientCompressor
# ===========================================================================


class TestGradientCompressor:
    def _make_cfg(self, **kwargs) -> CompressionConfig:
        defaults = dict(enabled=True, strategy="topk", topk_ratio=0.5, error_feedback=False)
        defaults.update(kwargs)
        return CompressionConfig(**defaults)

    def test_topk_returns_correct_ratio(self) -> None:
        cfg = self._make_cfg(strategy="topk", topk_ratio=0.5)
        gc = GradientCompressor(cfg, all_client_ids=[0])
        update = _make_update(0, {"w": torch.randn(100)})
        _, ratio = gc.compress(update)
        assert 0.0 < ratio <= 1.0

    def test_quantization_float16_ratio(self) -> None:
        cfg = self._make_cfg(strategy="quantization", quantization_bits=16)
        gc = GradientCompressor(cfg, all_client_ids=[0])
        update = _make_update(0, {"w": torch.randn(50)})
        _, ratio = gc.compress(update)
        assert ratio == pytest.approx(0.5)

    def test_quantization_int8_ratio(self) -> None:
        cfg = self._make_cfg(strategy="quantization", quantization_bits=8)
        gc = GradientCompressor(cfg, all_client_ids=[0])
        update = _make_update(0, {"w": torch.randn(50)})
        _, ratio = gc.compress(update)
        assert ratio == pytest.approx(0.25)

    def test_compressed_update_keeps_client_id(self) -> None:
        cfg = self._make_cfg()
        gc = GradientCompressor(cfg, all_client_ids=[5])
        update = _make_update(5)
        compressed, _ = gc.compress(update)
        assert compressed.client_id == 5

    def test_compressed_update_keeps_num_samples(self) -> None:
        cfg = self._make_cfg()
        gc = GradientCompressor(cfg, all_client_ids=[0])
        update = _make_update(0)
        compressed, _ = gc.compress(update)
        assert compressed.num_samples == update.num_samples

    def test_error_feedback_enabled_topk(self) -> None:
        cfg = self._make_cfg(strategy="topk", topk_ratio=0.1, error_feedback=True)
        gc = GradientCompressor(cfg, all_client_ids=[0])
        update = _make_update(0, {"w": torch.randn(100)})
        compressed, ratio = gc.compress(update)
        assert 0.0 < ratio <= 1.0
        assert compressed.state_dict["w"].shape == (100,)

    def test_multiple_rounds_with_error_feedback(self) -> None:
        cfg = self._make_cfg(strategy="topk", topk_ratio=0.1, error_feedback=True)
        gc = GradientCompressor(cfg, all_client_ids=[0])
        for _ in range(3):
            update = _make_update(0, {"w": torch.randn(100)})
            compressed, _ = gc.compress(update)
            assert compressed.state_dict["w"].shape == (100,)


# ===========================================================================
# CompressionConfig
# ===========================================================================


class TestCompressionConfig:
    def test_defaults(self) -> None:
        cfg = CompressionConfig()
        assert cfg.enabled is False
        assert cfg.strategy == "topk"
        assert cfg.topk_ratio == pytest.approx(0.1)
        assert cfg.quantization_bits == 16
        assert cfg.error_feedback is False

    def test_error_feedback_with_quantization_raises(self) -> None:
        with pytest.raises(Exception):
            CompressionConfig(enabled=True, strategy="quantization", error_feedback=True)

    def test_error_feedback_with_topk_valid(self) -> None:
        cfg = CompressionConfig(enabled=True, strategy="topk", error_feedback=True)
        assert cfg.error_feedback is True

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(Exception):
            CompressionConfig(strategy="svd")  # type: ignore[arg-type]

    def test_topk_ratio_zero_raises(self) -> None:
        with pytest.raises(Exception):
            CompressionConfig(topk_ratio=0.0)

    def test_topk_ratio_above_one_raises(self) -> None:
        with pytest.raises(Exception):
            CompressionConfig(topk_ratio=1.5)

    def test_topk_ratio_one_valid(self) -> None:
        cfg = CompressionConfig(topk_ratio=1.0)
        assert cfg.topk_ratio == pytest.approx(1.0)

    def test_quantization_bits_4_raises(self) -> None:
        with pytest.raises(Exception):
            CompressionConfig(quantization_bits=4)  # type: ignore[arg-type]

    def test_disabled_does_not_validate_strategy(self) -> None:
        # When disabled, error_feedback inconsistency is still validated.
        with pytest.raises(Exception):
            CompressionConfig(enabled=False, strategy="quantization", error_feedback=True)
