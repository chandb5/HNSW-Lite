import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hnsw-lite",
    version="0.1.0",
    author="Chand",
    author_email="chandbud5@gmail.com",
    description="A lightweight, pure Python implementation of HNSW algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chandb5/HNSW-Lite/hnsw-lite",
    project_urls={
        "Bug Tracker": "https://github.com/chandb5/HNSW-Lite/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=["plots", "plots.*", "test", "test.*", "venv", "venv.*"]),
    py_modules=["hnsw", "node", "distance", "search"],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
    ],
)