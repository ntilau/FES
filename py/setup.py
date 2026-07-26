from setuptools import setup, find_packages

setup(
    name="pyfes",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.4",
    ],
    extras_require={
        "ml": ["torch>=2.0", "gpytorch>=1.10", "scikit-learn>=1.0"],
        "viz": ["pyvista>=0.48"],
        "dev": ["pytest>=7.0", "torch>=2.0", "gpytorch>=1.10",
                "scikit-learn>=1.0", "pyvista>=0.48"],
    },
    python_requires=">=3.9",
    description=(
        "2D finite element solver for electromagnetics — "
        "waveguide, filter, circulator, and thermal simulation "
        "on triangular meshes, with DNN-GP surrogate modelling"
    ),
    keywords=(
        "fem, finite-elements, electromagnetics, waveguide, "
        "circulator, harmonic-balance, gaussian-process, "
        "gpytorch, surrogate-model, deep-kernel-learning, "
        "scipy, triangle-mesh"
    ),
    long_description=(
        __import__("pathlib").Path("README.md").read_text()
        if __import__("pathlib").Path("README.md").exists()
        else ""
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/ntilau/pyFES",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: Free for non-commercial use",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
