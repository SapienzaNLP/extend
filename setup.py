from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="extend",
    version="1.0.1",
    description="Extractive Entity Disambiguation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    url="https://github.com/sapienzanlp/extend",
    license="cc-by-nc-sa-4.0",
    package_data={"configurations": ["*.yaml", "*/*.yaml"]},
    install_requires=requirements,
    python_requires=">=3.8.0",
    author="Edoardo Barba",
    author_email="barba@di.uniroma1.it",
    zip_safe=False,
)
