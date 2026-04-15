from setuptools import setup, find_packages

setup(
    name="datapulse",
    version="1.0.0",
    description="Enterprise Analytics & ML Platform",
    author="DataPulse Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.15.0",
        "fastapi>=0.104.0",
        "pydantic>=2.0.0",
    ],
)
