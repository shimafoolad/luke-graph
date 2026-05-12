"""
setup.py for LUKE-Graph.

Install in development mode with:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="luke-graph",
    version="1.0.0",
    description=(
        "LUKE-Graph: A Transformer-based Approach with Gated Relational "
        "Graph Attention for Cloze-style Reading Comprehension"
    ),
    author="Shima Foolad, Kourosh Kiani",
    author_email="",
    url="https://github.com/studio-ousia/luke",
    license="Apache 2.0",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.10.0",
        "torch-geometric>=2.0.0",
        "click>=7.0",
        "tqdm>=4.62.0",
        "numpy>=1.21.0",
    ],
    entry_points={
        "console_scripts": [
            "luke-graph=luke_graph.main:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
