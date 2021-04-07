import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="optimizr",
    version="1.0.5",
    author="Joseph Riddle, Clayton Gravatt",
    author_email="joeriddles10@gmail.com",
    description="A small package to find similar collections.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joeriddles/optimizr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
