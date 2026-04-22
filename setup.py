from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="irsa",
    version="1.0.0",
    author="Your Name",
    author_email="your@email.com",
    description="Desktop application for radar image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/irsa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "irsa=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.pkl", "*.png", "*.txt"],
    },
    data_files=[
        ("matrices", ["matrices/*.pkl"]),
    ],
)