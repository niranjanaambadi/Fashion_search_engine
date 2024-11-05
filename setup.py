from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fashion-search-engine",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An intelligent fashion catalog system with classification and visual search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fashion-search-engine",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
        "api": [
            "fastapi>=0.95.0",
            "uvicorn>=0.22.0",
            "python-multipart>=0.0.6",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "fashion-train-classifier=train_classifier:main",
            "fashion-train-siamese=train_siamese:main",
            "fashion-search=search:main",
        ],
    },
)
