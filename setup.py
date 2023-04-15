#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

setup(
    name="LookoutX",
    version="0.0.0",
    description="Smart Assistant for the Visually Impaired",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Shreayan Chaudhary",
    author_email="shreayan98c@gmail.com",
    url="https://github.com/shreayan98c/LookoutX",
    install_requires=[
        "click",
        "rich",
        "numpy",
        "torch",
        "torchvision",
    ],
    packages=find_packages(),
)
