from setuptools import setup, find_packages

setup(
    name="mavrick",
    version="0.1.0",
    description="A Python image processing library using JAX",
    author="Nikhil Francis Giji",
    author_email="nikhilfrancisgiji@gmail.com",
    url="https://github.com/nikhilgiji/mavrick",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "numpy",
        "Pillow",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
