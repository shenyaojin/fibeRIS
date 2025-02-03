from setuptools import setup, find_packages

from setuptools import setup, find_packages

setup(
    name="fiberis",
    version="0.1.0",
    description="Fiber = Reservoir Integrated Simulator",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Shenyao Jin",
    author_email="shenyaojin@mines.edu",
    url="https://github.com/shenyaojin/fibeRIS",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy~=1.26.4",
        "matplotlib~=3.9.2",
        "scipy~=1.13.1",
        "pandas~=2.2.2",
        "h5py~=3.11.0",
        "plotly~=5.24.1",
        "python-dateutil~=2.9.0post0"
    ],
    extras_require={
        "dev": ["pytest", "sphinx", "black", "flake8", "mypy"],
        "docs": ["mkdocs", "mkdocs-material"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)