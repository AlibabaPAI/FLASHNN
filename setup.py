from pathlib import Path
from setuptools import find_packages, setup
from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="flashnn",
    version="0.1.1",
    description="Flash: Triton Kernel Library for LLM Serving",
    author="Alibaba PAI Team",
    url="https://github.com/AlibabaPAI/FLASHNN",
    packages=find_packages(),
    setup_requires=[
        'setuptools'
    ],
    install_requires=[
        "torch",
        "triton",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
