"""Setup script for Aura ML package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
def read_requirements(filename):
    req_file = Path(__file__).parent / "requirements" / filename
    if req_file.exists():
        with open(req_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("-r")]
    return []

setup(
    name="aura-ml",
    version="1.0.0",
    author="Aura ML Team",
    description="Emotional Support AI System powered by Llama 3.2 3B",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aura-ml",
    packages=find_packages(include=["aura_ml", "aura_ml.*", "api", "api.*", "cli", "cli.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("base.txt"),
    extras_require={
        "training": read_requirements("training.txt"),
        "api": read_requirements("api.txt"),
        "dev": read_requirements("dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "aura-chat=cli.chat:main",
        ],
    },
)
