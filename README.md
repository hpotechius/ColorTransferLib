![](https://img.shields.io/badge/Also%2C%20check%20out%20the%20Web%20Application%20ColorTransferLab%20at-red?style=flat-square)

https://github.com/hpotechius/ColorTransferLab and https://potechius.com/ColorTransferLab


![colortransferexample](https://github.com/user-attachments/assets/d5052bb6-e39b-445a-8abf-9966d02596d8)
# ColorTransferLib
![](https://img.shields.io/badge/ColorTransferLabV2-1.0.0-black?link=https%3A%2F%2Fgithub.com%2Fhpotechius%2FColorTransferLab) ![python3.12.6](https://img.shields.io/badge/build-3.12.6-blue?logo=python&label=Python) ![](https://img.shields.io/badge/build-24.04.3%20LTS-orange?logo=ubuntu&label=Ubuntu
) ![](https://img.shields.io/badge/build-MIT-purple?label=License) ![](https://img.shields.io/badge/build-GeForce%20RTX%204060%20Ti-white?logo=nvidia&label=GPU) ![](https://img.shields.io/badge/build-intel%20Core%20i7--14700KF-white?logo=intel&label=CPU) ![](https://img.shields.io/badge/macOS-Tahoe-deepskyblue?logo=apple) ![](https://img.shields.io/badge/Apple%20Silicon-M1-white)

---

ColorTransferLib is a library dedicated to color transfer, style transfer, and colorization, featuring a diverse range of published algorithms. Some methods have been re-implemented, while others are integrated from public repositories.

The primary goal of this project is to consolidate all existing color transfer, style transfer, and colorization techniques into a single library with a standardized API. This facilitates both the development and comparison of algorithms within the research community.

Currently, the library supports 11 color transfer, 5 style transfer, and 3 colorization methods across various data types, including images, point clouds, textured meshes, light fields, videos, volumetric videos, and Gaussian splatting. Additionally, it provides 20 evaluation metrics for assessing image-to-image color transfer performance.

A compatibility chart for supported data types and a detailed list of all algorithms can be found below.

![ColorTransferLabV2_DataTypes_wFiles](https://github.com/user-attachments/assets/3852256a-8547-4a36-be63-82f064d9f9b5)

## API
For seamless integration, adhere to the API specifications of the new color transfer algorithm, depicted in the Figure below.

Each class demands three inputs: Source, Reference, and Options. The Source and Reference should be of the **Image**, **Video**, **VolumetricVideo**, **LightField**, **GaussianSplatting** or **Mesh** class type, with the latter encompassing 3D point clouds and textured triangle meshes. The Options input consists of dictionaries, stored as a JSON file in the **Options** folder. For a sample option, see Listings 1. Every option details an adjustable parameter for the algorithm.

Save each new color transfer class in the ColorTransferLib Repository under the **Algorithms** folder. The class should have the **apply(...)** function, which ingests the inputs and embodies the core logic for color transfer.

The output should resemble a dictionary format, as outlined in Listing 2. A status code of 0 signifies a valid algorithm output, while -1 indicates invalidity. The process time denotes the algorithm's execution duration, useful for subsequent evaluations. The 'object' key in the dictionary holds the result, which should match the class type of the Source input.

![CT-API_new](https://github.com/user-attachments/assets/e52ed0ba-3106-435d-b7f1-fbac67145251)

<img alt="280272638-42e78a4f-89dc-4afe-876c-a1950044d514" src="https://github.com/user-attachments/assets/afde3c2a-a72f-4f9e-9be3-e5505faf46a7" />

## Installation

### Install via PyPI
```
pip install colortransferlib
pip install git+https://github.com/facebookresearch/detectron2.git@main
```

### Install from source
```
pip install -r requirements/requirements.txt -c requirements/constraints.txt
python setup.py bdist_wheel
pip install ../ColorTransferLib/dist/ColorTransferLib-2.1.0-py3-none-any.whl 
pip install git+https://github.com/facebookresearch/detectron2.git@main
```

## Usage
### Color Transfer
```python
from ColorTransferLib.ColorTransfer import ColorTransfer
from ColorTransferLib.DataTypes.Image import Image

src = Image(file_path='/media/source.png')
ref = Image(file_path='/media/reference.png') 

algo = "Reinhard01"
ct = ColorTransfer(src, ref, algo)
out = ct.apply()

# No output file extension has to be given
if out["status_code"] == 0:
    out["object"].write("/media/out")
else:
    print("Error: " + out["response"])
```

### Evaluation
```python
from ColorTransferLib.ColorTransfer import ColorTransferEvaluation
from ColorTransferLib.DataTypes.Image import Image

src = Image(file_path='/media/source.png')
ref = Image(file_path='/media/reference.png') 
out = Image(file_path='/media/output.png') 

cte = ColorTransferEvaluation(src, ref, out)
method = "CTQM"
eva = cte.apply(method)
print(eva)
```

## Test
```python
# Test all Color Transfer algorithms with all data type combinations
python main.py --test all_CT --out_path "/media/out"

# Test all Style Transfer algorithms with all data type combinations
python main.py --test all_ST --out_path "/media/out"

# Test all Colorization algorithms with all data type combinations
python main.py --test all_CT --out_path "/media/out"

# Test all evaluation metric on src, ref and out images
python main.py --test all_EVAL
```

## Available Methods:
The following color transfer, style transfer and colorization methods are integrated in the library. Some of them are reimplemented based on the algorithm's description in the the published papers and others are adopted from existing repositories and adpated to fit the API. The original implementation of the latter methods can be found next to the publication's name. Highlighted icon within the Support Column indicates the supported data types (From left to right: (1) Gaussian Splatting, (2) Light Field, (3) Volumetric Video, (4) Video, (5) Point Cloud, (6) Mesh, (7) Image)
### Color Transfer
| Year | ID  | Support |  Publication |
| ---  | --- | --- | --- |
| 2001 | $`Reinhard01`$ | <img src="https://github.com/user-attachments/assets/1589ba94-630a-420c-8309-09d01ea568ce" alt="Logo" style="width: 140px; height: 20px;"> | [Color Transfer between Images](https://doi.org/10.1109/38.946629) |
| 2003 | $`Chang03`$ | <img src="https://github.com/user-attachments/assets/1589ba94-630a-420c-8309-09d01ea568ce" alt="Logo" style="width: 140px; height: 20px;"> | [A Framework for Transfer Colors Based on the Basic Color Categories](https://doi.org/10.1109/CGI.2003.1214463) |
| 2005 | $`Pitie05`$ | <img src="https://github.com/user-attachments/assets/1589ba94-630a-420c-8309-09d01ea568ce" alt="Logo" style="width: 140px; height: 20px;"> | [N-dimensional probability density function transfer and its application to color transfer](https://doi.org/10.1109/ICCV.2005.166) |
| 2006 | $`Xiao06`$ | <img src="https://github.com/user-attachments/assets/1589ba94-630a-420c-8309-09d01ea568ce" alt="Logo" style="width: 140px; height: 20px;"> | [Color transfer in correlated color space](https://doi.org/10.1145/1128923.1128974) |
| 2007 | $`Reinhard07`$ | <img src="https://github.com/user-attachments/assets/1589ba94-630a-420c-8309-09d01ea568ce" alt="Logo" style="width: 140px; height: 20px;"> | [The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer](https://doi.org/10.1049/cp:20070055) |
| 2009 | $`Xiao09`$ | <img src="https://github.com/user-attachments/assets/f1d9165b-7a80-452b-b4bf-84cbaf4cef46" alt="Logo" style="width: 140px; height: 20px;"> | [Color Transfer between Images](https://doi.org/10.1109/38.946629) | [Gradient-Preserving Color Transfer](http://dx.doi.org/10.1111/j.1467-8659.2009.01566.x) |
| 2010 | $`Qian10`$ | <img src="https://github.com/user-attachments/assets/1589ba94-630a-420c-8309-09d01ea568ce" alt="Logo" style="width: 140px; height: 20px;"> | [An efficient fuzzy clustering-based color transfer method](https://doi.org/10.1109/FSKD.2010.5569560) |
| 2019 | $`Grogan19`$ | <img src="https://github.com/user-attachments/assets/f1d9165b-7a80-452b-b4bf-84cbaf4cef46" alt="Logo" style="width: 140px; height: 20px;"> | [L2 Divergence for robust colour transfer](https://doi.org/10.1016/j.cviu.2019.02.002) - [Original Implementation](https://github.com/groganma/gmm-colour-transfer) |
| 2020 | $`Lee20`$ | <img src="https://github.com/user-attachments/assets/f1d9165b-7a80-452b-b4bf-84cbaf4cef46" alt="Logo" style="width: 140px; height: 20px;"> | [Deep Color Transfer using Histogram Analogy](https://doi.org/10.1007/s00371-020-01921-6) - [Original Implementation](https://github.com/codeslake/Color_Transfer_Histogram_Analogy) |
| 2021 | $`Afifi21 _2`$ | <img src="https://github.com/user-attachments/assets/f1d9165b-7a80-452b-b4bf-84cbaf4cef46" alt="Logo" style="width: 140px; height: 20px;"> | [HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms](https://doi.org/10.48550/arXiv.2011.11731) |
| 2021 | $`Goude21`$ | <img src="https://github.com/user-attachments/assets/2c875956-1eaa-4ad4-bdfe-b2a0dd895b64" alt="Logo" style="width: 140px; height: 20px;"> | [Example-Based Colour Transfer for 3D Point Clouds](https://doi.org/10.1111/cgf.14388) |

### Style Transfer
| Year | ID | Support | Publication |
| ---  | --- | --- | --- |
| 2015 | $`Gatys15`$ | <img src="https://github.com/user-attachments/assets/f1d9165b-7a80-452b-b4bf-84cbaf4cef46" alt="Logo" style="width: 140px; height: 20px;"> | [A Neural Algorithm of Artistic Style](https://doi.org/10.48550/arXiv.1508.06576) - [Original Implementation](https://github.com/cysmith/neural-style-tf) |
| 2017 | $`Luan17`$ | <img src="https://github.com/user-attachments/assets/f1d9165b-7a80-452b-b4bf-84cbaf4cef46" alt="Logo" style="width: 140px; height: 20px;"> | [Deep Photo Style Transfer](https://doi.org/10.48550/arXiv.1703.07511) - [Original Implementation](https://github.com/LouieYang/deep-photo-styletransfer-tf) |
| 2020 | $`Cao20`$ | <img src="https://github.com/user-attachments/assets/2c875956-1eaa-4ad4-bdfe-b2a0dd895b64" alt="Logo" style="width: 140px; height: 20px;"> | [PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color](https://doi.org/10.1109/WACV45572.2020.9093513) - [Original Implementation](https://github.com/hoshino042/psnet) |
| 2021 | $`Afifi21`$ | <img src="https://github.com/user-attachments/assets/f1d9165b-7a80-452b-b4bf-84cbaf4cef46" alt="Logo" style="width: 140px; height: 20px;"> | [CAMS: Color-Aware Multi-Style Transfer](https://doi.org/10.48550/arXiv.2106.13920) - [Original Implementation](https://github.com/mahmoudnafifi/color-aware-style-transfer) |
| 2022 | $`Deng22`$ | <img src="https://github.com/user-attachments/assets/f1d9165b-7a80-452b-b4bf-84cbaf4cef46" alt="Logo" style="width: 140px; height: 20px;"> | [Stytr2: Image style transfer with transformers](https://doi.org/10.48550/arXiv.2105.14576) - [Original Implementation](https://github.com/diyiiyiii/StyTR-2) |

### Colorization
| Year | ID | Support | Publication |
| ---  | --- | --- | --- |
| 2020 | $`Su20`$ | <img src="https://github.com/user-attachments/assets/b2dd7598-3fe4-4d9a-abb5-8ee67bf64956" alt="Logo" style="width: 140px; height: 20px;"> | [Instance-aware image colorization](https://doi.org/10.48550/arXiv.2005.10825) - [Original Implementation](https://github.com/ericsujw/InstColorization) |
| 2022 | $`Ji22`$ | <img src="https://github.com/user-attachments/assets/b2dd7598-3fe4-4d9a-abb5-8ee67bf64956" alt="Logo" style="width: 140px; height: 20px;"> | [Colorformer: Image colorization via color memory assisted hybrid-attention transformer](https://doi.org/10.1007/978-3-031-19787-1_2) - [Original Implementation](https://github.com/jixiaozhong/ColorFormer) |
| 2023 | $`Kang23`$ | <img src="https://github.com/user-attachments/assets/b2dd7598-3fe4-4d9a-abb5-8ee67bf64956" alt="Logo" style="width: 140px; height: 20px;"> | [DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders](https://doi.org/10.48550/arXiv.2212.11613) - [Original Implementation](https://github.com/piddnad/DDColor) |

## Available Objective Evaluation Metrics
Three classes of evaluation metrics are considered here. Metrics that evaluate the color consistency with the reference image (indicated with $`^r`$), metrics that evaluate the structural similarity with the source image (indicated with $`^s`$) and metrics that evaluates the overall quality of the output (indicated with $`^o`$).

| Year | ID  | Name | Publication |
| ---  | --- | --- | --- |
| / | $`PSNR^s_{rgb}`$ | Peak Signal-to-Noise Ratio | / |
| / | $`HI^r_{rgb}`$ | Histogram Intersection | / |
| / | $`Corr^r_{rgb}`$ | Correlation | / |
| / | $`BD^r_{rgb}`$ | Bhattacharyya Distance | / |
| / | $`MSE^s_{rgb}`$ | Mean-Squared Error | / |
| / | $`RMSE^s_{rgb}`$ | Root-Mean-Squared Error | / |
| 2003 | $`CF^o_{rgyb}`$ | Colorfulness | [Measuring Colourfulness in Natural Images](http://dx.doi.org/10.1117/12.477378) |
| 2003 | $`MSSSIM^s_{rgb}`$ | Multi-Scale Structural Similarity Index | [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216) |
| 2004 | $`SSIM^s_{rgb}`$ | Structural Similarity Index | [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861) |
| 2006 | $`GSSIM^s_{rgb}`$ | Gradient-based Structural Similarity Index | [Gradient-Based Structural Similarity for Image Quality Assessment](https://doi.org/10.1109/ICIP.2006.313132) |
| 2010 | $`IVSSIM^s_{rgb}`$ | 4-component Structural Similarity Index | [Content-partitioned structural similarity index for image quality assessment](https://doi.org/10.1016/j.image.2010.03.004) |
| 2011 | $`IVEGSSIM^s_{rgb}`$ | 4-component enhanced Gradient-based Structural Similarity Index | [An image similarity measure using enhanced human visual system characteristics](https://ui.adsabs.harvard.edu/link_gateway/2011SPIE.8063E..10N/doi:10.1117/12.883301) |
| 2011 | $`FSIM^s_{c,yiq}`$ | Feature Similarity Index | [FSIM: A Feature Similarity Index for Image Quality Assessment](https://doi.org/10.1109/TIP.2011.2109730) |
| 2012 | $`BRISQUE^o_{rgb}`$ | Blind/Referenceless Image Spatial Quality Evaluator | [No-Reference Image Quality Assessment in the Spatial Domain](https://doi.org/10.1109/TIP.2012.2214050) |
| 2013 | $`NIQE^o_{rgb}`$ | Naturalness Image Quality Evaluator | [Making a “Completely Blind” Image Quality Analyzer](https://doi.org/10.1109/LSP.2012.2227726) |
| 2014 | $`VSI^s_{rgb}`$ | Visual Saliency-based Index | [VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment](https://doi.org/10.1109/TIP.2014.2346028) |
| 2016 | $`CTQM^{sro}_{lab}`$ | Color Transfer Quality Metric | [Novel multi-color transfer algorithms and quality measure](https://doi.org/10.1109/TCE.2016.7613196) |
| 2018 | $`LPIPS^s_{rgb}`$ | Learned Perceptual Image Patch Similarity | [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://doi.org/10.1109/CVPR.2018.00068) |
| 2018 | $`NIMA^o_{rgb}`$ | Neural Image Assessment | [NIMA: Neural Image Assessment](https://doi.org/10.48550/arXiv.1709.05424) |
| 2019 | $`CSS^{sr}_{rgb}`$ | Color and Structure Similarity | [Selective color transfer with multi-source images](https://doi.org/10.1016/j.patrec.2009.01.004) |

## Issues
- PSN crashes if the point clouds are too large
- NIMA is not working on Mac OS

## Citation
If you utilize this code in your research, kindly provide a citation:
```
@article{potechius2023,
  author={Herbert Potechius, Thomas Sikora, Gunasekaran Raja, Sebastian Knorr},
  title={A software test bed for sharing and evaluating color transfer algorithms for images and 3D objects},
  year={2023},
  booktitle={European Conference on Visual Media Production (CVMP)},
  doi={10.1145/3626495.3626509}
}
```
```
@article{potechius2025,
  author = {Potechius,  Herbert and Sikora,  Thomas and Knorr, Sebastian},
  title = {ColorTransferLabV2: a software testbed for multi-modal color transfer, colorization, and style transfer},
  journal = {Journal of Electronic Imaging},
  editor = {SPIE},
  year = {2025},
  volume = {34},
  issue = {5},
  pages = {1-31},
  doi = {10.1117/1.JEI.34.5.051002},
  url = { https://doi.org/10.1117/1.JEI.34.5.051002}
}
```
