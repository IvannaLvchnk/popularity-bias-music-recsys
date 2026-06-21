from setuptools import setup, find_packages

setup(
    name="popbias",
    version="1.0.0",
    description="Popularity Bias Measurement & Mitigation Toolkit for Music Recommender Systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    authors=[
        {"name": "Ivanna Levchenko", "email": "h12200708@wu.ac.at"},
        {"name": "Oleksandr Ursol",  "email": "h12438168@wu.ac.at"},
    ],
    url="https://github.com/YOUR_USERNAME/popularity-bias-music-recsys",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "implicit>=0.7",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "matplotlib>=3.7",
            "seaborn>=0.12",
            "pyarrow>=12.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
