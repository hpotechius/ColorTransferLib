import setuptools

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Reading the content of the requirements.txt
with open('requirements/requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="ColorTransferLib",
    version="2.2.0",
    author="Herbert Potechius",
    author_email="herbert@potechius.com",
    description="This library provides color and tyle transfer algorithms which were published in scientific papers. Additionall a set of IQA metrics are available.",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    package_data={"": ['Options/*.json',
                       'Algorithms/*/*.json',
                       'Config/*.json', 
                       'Evaluation/VSI/third_party/saliency_models/resources/*.mat',
                       'Algorithms/Kang23/third_party/basicsr/archs/ddcolor_arch_utils/*']},
    include_package_data=True,
    python_requires='>=3.12,<3.13',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements
)