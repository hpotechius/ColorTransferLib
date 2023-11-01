![colortransfer_example](https://github.com/ImmersiveMediaLaboratory/ColorTransferLib/assets/15614886/928791b0-b734-4835-92c9-cdcb12fcddc7)
# ColorTransferLib
The ColorTransferLib is a library focused on color transfer, featuring a range of published algorithms. Some algorithms have been re-implemented, while others are integrated from public repositories. The primary objective of this project is to compile all existing color and style transfer methods into one library with a standardized API. This aids the research community in both development and comparison of algorithms. Currently, the library supports 15 color and style transfer methods for images, 3D point clouds, and textured triangle meshes. Additionally, it includes 20 metrics for evaluating color transfer results. A detailed list of all algorithms is available below.

![compatability](https://github.com/ImmersiveMediaLaboratory/ColorTransferLib/assets/15614886/d59972fd-e135-4572-8682-d3f5f8f85c75)

## API
For seamless integration, adhere to the API specifications of the new color transfer algorithm, depicted in the Figure below.

Each class demands three inputs: Source, Reference, and Options. The Source and Reference should be of the **Image** or **Mesh** class type, with the latter encompassing 3D point clouds and textured triangle meshes. The Options input consists of dictionaries, stored as a JSON file in the **Options** folder. For a sample option, see Listings 1. Every option details an adjustable parameter for the algorithm.

Save each new color transfer class in the ColorTransferLib Repository under the **Algorithms** folder. This ensures its automatic integration into the user interface. The class should have two essential functions: **get_info()** and **apply(...)**. The **get_info()** function yields vital details about the algorithm (refer to Listing 2). It also provides data type details, facilitating the identification of compatible objects for the algorithm. The **apply(...)** function ingests the inputs and embodies the core logic for color transfer.

The output should resemble a dictionary format, as outlined in Listing 3. A status code of 0 signifies a valid algorithm output, while -1 indicates invalidity. The process time denotes the algorithm's execution duration, useful for subsequent evaluations. The 'object' key in the dictionary holds the result, which should match the class type of the Source input.

![CT-API](https://github.com/ImmersiveMediaLaboratory/ColorTransferLib/assets/15614886/7e59eea8-78be-4dfb-acae-7e8abfd7abe5)

![Listing](https://github.com/ImmersiveMediaLaboratory/ColorTransferLib/assets/15614886/93692741-46f6-4955-80a5-d152fa22104d)

## Installation
``
python setup.py bdist_wheel
``

``
pip install --force-reinstall ../ColorTransferLib/dist/ColorTransferLib-0.0.2-py3-none-any.whl 
``

Necessary for TpsColorTransfer
``
sudo apt-get install octave
``

Necessary for installing dlib -> HistoGAN
``
sudo apt-get install python3-dev
``

Instalation of pyfftw
``
(env-CTL) potechius@HMP-MacBook ColorTransferLib % export DYLD_LIBRARY_PATH=/opt/homebrew/lib export LDFLAGS="-L/opt/homebrew/lib"
(env-CTL) potechius@HMP-MacBook ColorTransferLib % export CFLAGS="-I/opt/homebrew/include"
``

run: for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
to get rid of the info: successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero



## Available Color Transfer Methods:
The following color transfer methods are integrated in the library. Some of them are reimplemented based on the algorithm's description in the the published papers and others are adopted from existing repositories and adpated to fit the API. The original implementation of the latter methods can be found next to the **Source** entry.

| Year | ID  | Publication |
| ---  | --- | --- |
| 2001 | GLO$`^{2D,3D}_{CT}`$ | [Color Transfer between Images](https://doi.org/10.1109/38.946629) |
| 2003 | BCC$`^{2D,3D}_{CT}`$ | [A Framework for Transfer Colors Based on the Basic Color Categories](https://doi.org/10.1109/CGI.2003.1214463) |
| 2005 | PDF$`^{2D,3D}_{CT}`$ | [N-dimensional probability density function transfer and its application to color transfer](https://doi.org/10.1109/ICCV.2005.166) |
| 2006 | CCS$`^{2D,3D}_{CT}`$ | [Color transfer in correlated color space](https://doi.org/10.1145/1128923.1128974) |
| 2007 | MKL$`^{2D,3D}_{CT}`$ | [The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer](https://doi.org/10.1049/cp:20070055) |
| 2009 | GPC$`^{2D,3D}_{CT}`$ | [Gradient-Preserving Color Transfer](http://dx.doi.org/10.1111/j.1467-8659.2009.01566.x) |
| 2010 | FUZ$`^{2D,3D}_{CT}`$ | [An efficient fuzzy clustering-based color transfer method](https://doi.org/10.1109/FSKD.2010.5569560) |
| 2015 | NST$`^{2D,3D}_{ST}`$ | [A Neural Algorithm of Artistic Style](https://doi.org/10.48550/arXiv.1508.06576) - [Original Implementation](https://github.com/cysmith/neural-style-tf) |
| 2017 | DPT$`^{2D}_{ST}`$ | [Deep Photo Style Transfer](https://doi.org/10.48550/arXiv.1703.07511) - [Original Implementation](https://github.com/LouieYang/deep-photo-styletransfer-tf) |
| 2019 | TPS$`^{2D,3D}_{CT}`$ | [L2 Divergence for robust colour transfer](https://doi.org/10.1016/j.cviu.2019.02.002) - [Original Implementation](https://github.com/groganma/gmm-colour-transfer) |
| 2020 | HIS$`^{2D,3D}_{CT}`$ | [Deep Color Transfer using Histogram Analogy](https://doi.org/10.1007/s00371-020-01921-6) - [Original Implementation](https://github.com/codeslake/Color_Transfer_Histogram_Analogy) |
| 2020 | PSN$`^{3D}_{ST}`$ | [PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color](https://doi.org/10.1109/WACV45572.2020.9093513) - [Original Implementation](https://github.com/hoshino042/psnet) |
| 2020 | EB3$`^{3D}_{CT}`$ | [Example-Based Colour Transfer for 3D Point Clouds](https://doi.org/10.1111/cgf.14388) |
| 2021 | CAM$`^{2D,3D}_{ST}`$ | [CAMS: Color-Aware Multi-Style Transfer](https://doi.org/10.48550/arXiv.2106.13920) - [Original Implementation](https://github.com/mahmoudnafifi/color-aware-style-transfer) |
| 2021 | RHG$`^{2D,3D}_{CT}`$ | [HistoGAN: Controlling Colors of GAN-Generated and Real Images via Color Histograms](https://doi.org/10.48550/arXiv.2011.11731) |


## Available Objective Evaluation Metrics
Three classes of evaluation metrics are considered here. Metrics that evaluate the color consistency with the reference image (indicated with $`^r`$), metrics that evaluate the structural similarity with the source image (indicated with $`^s`$) and metrics that evaluates the overall quality of the output (indicated with $`^o`$).

| Year | ID  | Name | Publication |
| ---  | --- | --- | --- |
| ... | PSNR$`^s_{rgb}`$ | Peak Signal-to-Noise Ratio | []() |
| 2004 | SSIM$`^s_{rgb}`$ | Structural Similarity Index | [Image quality assessment: from error visibility to structural similarity](https://doi.org/10.1109/TIP.2003.819861) |
| ... | MSE$`^s_{rgb}`$ | Mean-Squared Error | []() |
| 2003 | MS-SSIM$`^s_{rgb}`$ | Multi-Scale Structural Similarity Index | [Multiscale structural similarity for image quality assessment](https://doi.org/10.1109/ACSSC.2003.1292216) |
| ... | FSIM$`^s_{c,yiq}`$ | Feature Similarity Index | []() |
| ... | LPIPS$`^s_{rgb}`$ | Learned Perceptual Image Patch Similarity | []() |
| ... | RMSE$`^s_{rgb}`$ | Root-Mean-Squared Error | []() |
| 2006 | GSSIM$`^s_{rgb}`$ | Gradient-based Structural Similarity Index | [Gradient-Based Structural Similarity for Image Quality Assessment](https://doi.org/10.1109/ICIP.2006.313132) |
| ... | VSI$`^s_{rgb}`$ | Visual Saliency-based Index | []() |
| 2010 | 4-SSIM$`^s_{rgb}`$ | 4-component Structural Similarity Index | [Content-partitioned structural similarity index for image quality assessment](https://doi.org/10.1016/j.image.2010.03.004) |
| ... | HI$`^r_{rgb}`$ | Histogram Intersection | []() |
| ... | Corr$`^r_{rgb}`$ | Correlation | []() |
| ... | BA$`^r_{rgb}`$ | Bhattacharyya Distance | []() |
| 2012 | BRISQUE$`^o_{rgb}`$ | Blind/Referenceless Image Spatial Quality Evaluator | [No-Reference Image Quality Assessment in the Spatial Domain](https://doi.org/10.1109/TIP.2012.2214050) |
| 2018 | NIMA$`^o_{rgb}`$ | Neural Image Assessment | [NIMA: Neural Image Assessment](https://doi.org/10.48550/arXiv.1709.05424) |
| 2013 | NIQE$`^o_{rgb}`$ | Naturalness Image Quality Evaluator | [Making a “Completely Blind” Image Quality Analyzer](https://doi.org/10.1109/LSP.2012.2227726) |
| ... | CF$`^o_{rgyb}`$ | Colorfulness | []() |
| 2011 | 4-EGSSIM$`^s_{rgb}`$ | 4-component enhanced Gradient-based Structural Similarity Index | [An image similarity measure using enhanced human visual system characteristics](https://ui.adsabs.harvard.edu/link_gateway/2011SPIE.8063E..10N/doi:10.1117/12.883301) |
| ... | CSS$`^{sr}_{rgb}`$ | Color and Structure Similarity | []() |
| ... | CTQM$`^{sro}_{lab}`$ | Color Transfer Quality Metric | []() |

## Citation
If you utilize this code in your research, kindly provide a citation:
```
@inproceeding{potechius2023,
  title={A software test bed for sharing and evaluating color transfer algorithms for images and 3D objects},
  author={Herbert Potechius, Thomas Sikora, Gunasekaran Raja, Sebastian Knorr},
  year={2023},
  booktitle={European Conference on Visual Media Production (CVMP)},
  doi={10.1145/3626495.3626509}
}
```
