from setuptools import setup, find_packages

setup(
    name="qlens",
    version="0.1.0",
    description=(
        "Q-LENS: A Pre-Registered Probing Stress-Test of "
        "Vision-Language Quantitative-Physics Reasoning."
    ),
    author="Anonymous",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.45.0",
        "peft>=0.10.0",
        "accelerate>=0.30.0",
        "h5py>=3.11.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
        "huggingface-hub>=0.22.0",
    ],
    extras_require={
        "dev": ["pytest>=8.1.0", "pytest-cov>=5.0.0", "jupyter>=1.0.0"],
        "viz": ["plotly>=5.20.0", "opencv-python>=4.9.0"],
        "tracking": ["wandb>=0.17.0"],
    },
)
