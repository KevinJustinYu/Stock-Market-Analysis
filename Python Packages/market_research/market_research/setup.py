import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='market_research',
    version='0.0.0',
    author='Kevin Yu',
    author_email='kjyu@caltech.edu',
    description='Utilities for analyzing and obtaining data for stocks.',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    include_package_data=True
)