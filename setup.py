"""setup.py for the stimulator package."""

from pathlib import Path
from typing import Union

from setuptools import find_packages, setup


def parse_requirements(filename: Union[str, Path]) -> list[str]:
    """Parse a requirements file and return a list of requirements."""
    return Path(filename).read_text().splitlines()


setup(
    name="stimulator",
    version="0.1.0",
    description="Multimodal persona-aware conversational AI",
    author="Shon Verch",
    author_email="verchshon@gmail.com",
    license="MIT",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": parse_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "stimulator=stimulator.cli:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
