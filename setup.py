from setuptools import find_packages, setup

setup(
    name="flashnn",
    version="0.1.0",
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
)
