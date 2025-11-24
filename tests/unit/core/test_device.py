"""
Unit tests for device resolution helpers.
"""

from unittest.mock import MagicMock, patch

import pytest

from antibody_training_esm.core.device import _mps_available, resolve_device


class TestMpsAvailable:
    """Test _mps_available helper function."""

    @pytest.mark.unit
    def test_mps_available_when_supported(self) -> None:
        """Test MPS is detected when torch.backends.mps exists and is available."""
        with patch("torch.backends.mps") as mock_mps:
            mock_mps.is_available.return_value = True
            assert _mps_available() is True

    @pytest.mark.unit
    def test_mps_not_available_when_not_supported(self) -> None:
        """Test MPS is not detected when torch.backends.mps.is_available() is False."""
        with patch("torch.backends.mps") as mock_mps:
            mock_mps.is_available.return_value = False
            assert _mps_available() is False

    @pytest.mark.unit
    def test_mps_not_available_when_backend_missing(self) -> None:
        """Test MPS is not detected when torch.backends.mps doesn't exist."""
        # Mock torch.backends without mps attribute
        mock_backends = MagicMock(spec=[])  # spec=[] means no attributes
        with patch("torch.backends", mock_backends):
            assert _mps_available() is False


class TestResolveDeviceAuto:
    """Test resolve_device with 'auto' mode (and priority logic)."""

    @pytest.mark.unit
    def test_auto_detects_cuda_first(self) -> None:
        """Test auto mode prefers CUDA when available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "antibody_training_esm.core.device._mps_available", return_value=True
            ),
        ):
            assert resolve_device("auto") == "cuda"

    @pytest.mark.unit
    def test_auto_detects_mps_when_no_cuda(self) -> None:
        """Test auto mode uses MPS when CUDA unavailable."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "antibody_training_esm.core.device._mps_available", return_value=True
            ),
        ):
            assert resolve_device("auto") == "mps"

    @pytest.mark.unit
    def test_auto_falls_back_to_cpu(self) -> None:
        """Test auto mode falls back to CPU when no GPU available."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "antibody_training_esm.core.device._mps_available", return_value=False
            ),
        ):
            assert resolve_device("auto") == "cpu"

    @pytest.mark.unit
    def test_none_device_behaves_like_auto(self) -> None:
        """Test None device is treated same as 'auto'."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "antibody_training_esm.core.device._mps_available", return_value=False
            ),
        ):
            assert resolve_device(None) == "cpu"


class TestResolveDeviceExplicit:
    """Test resolve_device with explicit device requests."""

    @pytest.mark.unit
    def test_cpu_always_works(self) -> None:
        """Test CPU device always succeeds."""
        assert resolve_device("cpu") == "cpu"

    @pytest.mark.unit
    def test_cuda_succeeds_when_available(self) -> None:
        """Test CUDA device succeeds when available."""
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("cuda") == "cuda"

    @pytest.mark.unit
    def test_mps_succeeds_when_available(self) -> None:
        """Test MPS device succeeds when available."""
        with patch(
            "antibody_training_esm.core.device._mps_available", return_value=True
        ):
            assert resolve_device("mps") == "mps"


class TestResolveDeviceErrors:
    """Test resolve_device error handling."""

    @pytest.mark.unit
    def test_unknown_device_raises_value_error(self) -> None:
        """Test unknown device string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device 'gpu'"):
            resolve_device("gpu")

    @pytest.mark.unit
    def test_invalid_device_shows_expected_values(self) -> None:
        """Test error message shows expected device values."""
        with pytest.raises(
            ValueError, match="Expected one of: cpu, cuda, mps, or auto"
        ):
            resolve_device("tpu")

    @pytest.mark.unit
    def test_cuda_error_message_provides_guidance(self) -> None:
        """Test CUDA error message provides actionable guidance."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            pytest.raises(
                RuntimeError,
                match=r"Install a CUDA-enabled PyTorch build.*choose hardware\.device=cpu",
            ),
        ):
            resolve_device("cuda")

    @pytest.mark.unit
    def test_mps_error_message_provides_guidance(self) -> None:
        """Test MPS error message provides actionable guidance."""
        with (
            patch(
                "antibody_training_esm.core.device._mps_available", return_value=False
            ),
            pytest.raises(
                RuntimeError,
                match=r"Use hardware\.device=cpu.*install a PyTorch build with MPS support",
            ),
        ):
            resolve_device("mps")

    @pytest.mark.unit
    def test_malformed_input_raises_value_error(self) -> None:
        """Test malformed input (empty string, whitespace) raises ValueError."""
        # Empty string
        with pytest.raises(ValueError, match="Unknown device ''"):
            resolve_device("")

        # Whitespace
        with pytest.raises(ValueError, match="Unknown device ' cuda '"):
            resolve_device(" cuda ")


class TestResolveDeviceRealWorld:
    """Test resolve_device with real PyTorch state (current machine)."""

    @pytest.mark.unit
    def test_resolve_device_with_real_pytorch(self) -> None:
        """Test resolve_device works with actual PyTorch installation."""
        # This test uses real torch.cuda.is_available() and torch.backends.mps
        # It should not raise errors regardless of platform
        result = resolve_device("auto")
        assert result in {"cpu", "cuda", "mps"}

    @pytest.mark.unit
    def test_cpu_always_works_real(self) -> None:
        """Test CPU device always works with real PyTorch."""
        assert resolve_device("cpu") == "cpu"

    @pytest.mark.unit
    def test_auto_and_none_equivalent_real(self) -> None:
        """Test 'auto' and None produce same result with real PyTorch."""
        assert resolve_device("auto") == resolve_device(None)
