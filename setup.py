import pathlib
from setuptools import setup, find_packages

this_directory = pathlib.Path(__file__).parent

with open(this_directory / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(this_directory / "requirements.txt") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith('#')]

setup(
    name="glimmer",
    version="0.2",
    author="Immanuel Weber",
    author_email="immanuel.weber@gmail.com",
    description="glimmer utilities library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
