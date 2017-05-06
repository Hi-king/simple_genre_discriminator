# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.md").read()
except IOError:
    long_description = ""


install_requires = [
]

setup(
    name="simple_genre_discriminator",
    version="0.1.0",
    description="collect anime images from miscellaneous images in the web",
    license="MIT",
    author="Hi_king",
    packages=find_packages(),
    install_requires=[],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
    ]
)
