import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TRMPINT", # Replace with your own username - name of package - called TRMPINT
    version="0.0.1",
    author="Evelyn Baker",
    author_email="eb2015@ic.ac.uk",
    description="TRMP package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject", #link to github page - need to make account ready for this bit
    packages=setuptools.find_packages(), #auto pick packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)