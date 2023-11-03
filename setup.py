import setuptools

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Reading the content of the requirements.txt
with open('requirements/requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="ColorTransferLib",
    version="1.0.0-2",
    author="Herbert Potechius",
    author_email="potechius.herbert@gmail.com",
    description="This library provides color and tyle transfer algorithms which were published in scientific papers. Additionall a set of IQA metrics are available.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    package_data={"": ['Options/*.json',
                       'Config/*.json',
                       'Algorithms/TPS/L2RegistrationForCT/*',
                       'Algorithms/TPS/L2RegistrationForCT/*/*',
                       'Algorithms/TPS/L2RegistrationForCT/*/*/*', 
                       'Evaluation/VIS/gbvs/*', 
                       'Evaluation/VIS/gbvs/*/*', 
                       'Evaluation/VIS/gbvs/*/*/*']},
    include_package_data=True,
    python_requires='>=3.8',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
)