from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
readme = (this_dir / "README.md").read_text(encoding="utf-8") if (this_dir / "README.md").exists() else ""

setup(
    name="tensorweave",
    version="0.2.0",  # bump as you go
    description="Interpolating geophysical tensor fields using spatial neural networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Akshay Kamath",
    url="https://github.com/k4m4th/tensorweave",   # optional
    license="MIT",                                 # or what you use
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.9",
        "pandas>=1.3",
        "torch>=2.0",
        "tqdm>=4.60",
        "matplotlib>=3.5",
    ],
    extras_require={
        "dev": ["black>=24.0", "ruff>=0.4", "mypy>=1.5", "pytest>=7.0"],
        "docs": ["sphinx>=7.0", "furo>=2024.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
)
