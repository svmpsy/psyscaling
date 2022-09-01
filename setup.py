from setuptools import setup, find_packages
import setuptools

#import pathlib
#here = pathlib.Path(__file__).parent.resolve()

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "nbconvert>=5", "requests>=2"]

setup(
    name="psyscaling",
    version="0.0.1",
    author="Svetlana Morozova",
    author_email="svmpsy1@gmail.com",
    description="Psychological scaling of text and graphic information. Used to prepare experimental data for quantitative analysis.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/svmpsy/psyscaling.git",
    packages=setuptools.find_packages(),
    install_requires=['pymorphy2', 're', 'nltk'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: MIT License",
    ],
       
    data_files=[("sensdict.csv", ["dict/sensdict.csv"])],
    
    python_requires='>=3.7',
)
