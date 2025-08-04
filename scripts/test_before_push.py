import pytest
import os
import subprocess
from pathlib import Path


class TestBeforePush:
    """Test suite to ensure basic functionality works before pushing code."""

    @pytest.fixture(autouse=True)
    def setup_datasets(self):
        """Download required mini datasets if they don't exist."""
        if not Path("./datasets/mini").exists():
            subprocess.run(["bash", "./datasets/download_cyclegan_dataset.sh", "mini"], check=True)
        
        if not Path("./datasets/mini_pix2pix").exists():
            subprocess.run(["bash", "./datasets/download_cyclegan_dataset.sh", "mini_pix2pix"], check=True)
        
        if not Path("./datasets/mini_colorization").exists():
            subprocess.run(["bash", "./datasets/download_cyclegan_dataset.sh", "mini_colorization"], check=True)

    def test_pretrained_cyclegan_model(self):
        """Test pretrained CycleGAN model download and inference."""
        if not Path("./checkpoints/horse2zebra_pretrained/latest_net_G.pth").exists():
            subprocess.run(["bash", "./scripts/download_cyclegan_model.sh", "horse2zebra"], check=True)
        
        result = subprocess.run([
            "python", "test.py", "--model", "test", "--dataroot", "./datasets/mini",
            "--name", "horse2zebra_pretrained", "--no_dropout", "--num_test", "1"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"CycleGAN test failed: {result.stderr}"

    def test_pretrained_pix2pix_model(self):
        """Test pretrained pix2pix model download and inference."""
        if not Path("./checkpoints/facades_label2photo_pretrained/latest_net_G.pth").exists():
            subprocess.run(["bash", "./scripts/download_pix2pix_model.sh", "facades_label2photo"], check=True)
        
        if not Path("./datasets/facades").exists():
            subprocess.run(["bash", "./datasets/download_pix2pix_dataset.sh", "facades"], check=True)
        
        result = subprocess.run([
            "python", "test.py", "--dataroot", "./datasets/facades/", "--direction", "BtoA",
            "--model", "pix2pix", "--name", "facades_label2photo_pretrained", "--num_test", "1"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0, f"Pix2pix test failed: {result.stderr}"

    def test_cyclegan_train_test(self):
        """Test CycleGAN training and testing pipeline."""
        # Train
        train_result = subprocess.run([
            "python", "train.py", "--model", "cycle_gan", "--name", "temp_cyclegan",
            "--dataroot", "./datasets/mini", "--n_epochs", "1", "--n_epochs_decay", "0",
            "--save_latest_freq", "10", "--print_freq", "1"
        ], capture_output=True, text=True)
        
        assert train_result.returncode == 0, f"CycleGAN training failed: {train_result.stderr}"
        
        # Test
        test_result = subprocess.run([
            "python", "test.py", "--model", "test", "--name", "temp_cyclegan",
            "--dataroot", "./datasets/mini", "--num_test", "1", "--model_suffix", "_A", "--no_dropout"
        ], capture_output=True, text=True)
        
        assert test_result.returncode == 0, f"CycleGAN testing failed: {test_result.stderr}"

    def test_pix2pix_train_test(self):
        """Test pix2pix training and testing pipeline."""
        # Train
        train_result = subprocess.run([
            "python", "train.py", "--model", "pix2pix", "--name", "temp_pix2pix",
            "--dataroot", "./datasets/mini_pix2pix", "--n_epochs", "1", "--n_epochs_decay", "5",
            "--save_latest_freq", "10"
        ], capture_output=True, text=True)
        
        assert train_result.returncode == 0, f"Pix2pix training failed: {train_result.stderr}"
        
        # Test
        test_result = subprocess.run([
            "python", "test.py", "--model", "pix2pix", "--name", "temp_pix2pix",
            "--dataroot", "./datasets/mini_pix2pix", "--num_test", "1"
        ], capture_output=True, text=True)
        
        assert test_result.returncode == 0, f"Pix2pix testing failed: {test_result.stderr}"

    def test_template_train_test(self):
        """Test template model training and testing."""
        # Train
        train_result = subprocess.run([
            "python", "train.py", "--model", "template", "--name", "temp2",
            "--dataroot", "./datasets/mini_pix2pix", "--n_epochs", "1", "--n_epochs_decay", "0",
            "--save_latest_freq", "10"
        ], capture_output=True, text=True)
        
        assert train_result.returncode == 0, f"Template training failed: {train_result.stderr}"
        
        # Test
        test_result = subprocess.run([
            "python", "test.py", "--model", "template", "--name", "temp2",
            "--dataroot", "./datasets/mini_pix2pix", "--num_test", "1"
        ], capture_output=True, text=True)
        
        assert test_result.returncode == 0, f"Template testing failed: {test_result.stderr}"

    def test_colorization_train_test(self):
        """Test colorization model training and testing."""
        # Train
        train_result = subprocess.run([
            "python", "train.py", "--model", "colorization", "--name", "temp_color",
            "--dataroot", "./datasets/mini_colorization", "--n_epochs", "1", "--n_epochs_decay", "0",
            "--save_latest_freq", "5"
        ], capture_output=True, text=True)
        
        assert train_result.returncode == 0, f"Colorization training failed: {train_result.stderr}"
        
        # Test
        test_result = subprocess.run([
            "python", "test.py", "--model", "colorization", "--name", "temp_color",
            "--dataroot", "./datasets/mini_colorization", "--num_test", "1"
        ], capture_output=True, text=True)
        
        assert test_result.returncode == 0, f"Colorization testing failed: {test_result.stderr}"
