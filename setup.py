import setuptools

setuptools.setup(
    name="ColorTransferLib",
    version="0.0.2",
    author="Herbert Potechius",
    author_email="potechius.herbert@gmail.com",
    description="This library provides color transfer algorithms which were published in scientific papers.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
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
)