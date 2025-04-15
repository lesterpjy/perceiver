#!/usr/bin/env python
from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="perceiver-io",
        version="0.11.1",
        description="Perceiver IO",
        author="Martin Krasser, Christoph Stumpf",
        packages=find_packages(),
        python_requires=">=3.8,<3.11",
    )
