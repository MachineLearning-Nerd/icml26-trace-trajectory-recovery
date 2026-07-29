import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.md file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="trace_crl",
    version="0.1.0",
    description="TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pytorch-lightning>=2.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy",
        "scipy",
        "matplotlib",
        "pyyaml",
        "ipdb",
        "h5py",
        "opencv-python",
        "pymunk",
        "scikit-learn",
    ],
)
