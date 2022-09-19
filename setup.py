from distutils.core import setup
from platform import platform
from setuptools import find_packages


setup(
    name='pixel',
    version=1.0,
    install_requires=[
        'cloudpickle',
        'numpy',
        'wandb',
        'gym[all]'
    ],
    description="Agnet-Pixel, Deep RL tools for pixel-based environments.",
    authors="Rami Ahmed",
    url="https://github.com/RamiSketcher/AgentPixel",
    author_email="ramisketcher@gmail.com"
)
