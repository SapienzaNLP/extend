from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="extend",
    version="0.1",
    packages=["extend"],
    url="",
    license="",
    install_requires=requirements,
    author="Edoardo Barba",
    author_email="barba@di.uniroma1.it",
    description="Extractive Entity Disambiguation",
)
