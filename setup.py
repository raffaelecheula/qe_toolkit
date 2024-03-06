from setuptools import setup, find_packages

setup(
    name="qe_toolkit",
    version="0.0.1",
    url="https://github.com/raffaelecheula/qe_toolkit.git",
    author="Raffaele Cheula",
    author_email="cheula.raffaele@gmail.com",
    description="Utilities for Quantum Espresso calculations.",
    license="GPL-3.0",
    install_requires=find_packages(),
    python_requires=">=3.5, <4",
)
