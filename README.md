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
| 2001 | GLO | [Color Transfer between Images](https://doi.org/10.1109/38.946629) |
| 2005 | PDF | [N-dimensional probability density function transfer and its application to color transfer](https://doi.org/10.1109/ICCV.2005.166) |
| 2007 | MKL | [The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer](https://doi.org/10.1049/cp:20070055) |
| 2010 | FUZ | [An efficient fuzzy clustering-based color transfer method](https://doi.org/10.1109/FSKD.2010.5569560) |
| 2015 | NST | [A Neural Algorithm of Artistic Style](https://doi.org/10.48550/arXiv.1508.06576) |
| 2017 | DPT | [Deep Photo Style Transfer](https://doi.org/10.48550/arXiv.1703.07511) |
| 2019 | TPS | [L2 Divergence for robust colour transfer](https://doi.org/10.1016/j.cviu.2019.02.002) |
| 2020 | HIS | [Deep Color Transfer using Histogram Analogy](https://doi.org/10.1007/s00371-020-01921-6) |
| 2020 | PSN | [PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color](https://doi.org/10.1109/WACV45572.2020.9093513) |
| 2020 | EB3 | [Example-Based Colour Transfer for 3D Point Clouds](https://doi.org/10.1111/cgf.14388) |
| 2021 | CAM | [CAMS: Color-Aware Multi-Style Transfer](https://doi.org/10.48550/arXiv.2106.13920) |
| 2003 | BCC | [A Framework for Transfer Colors Based on the Basic Color Categories](https://doi.org/10.1109/CGI.2003.1214463) |
| XXX | RHG | []() |
| XXX | CCS | []() |
| XXX | GPC | []() |


## Available Objective Evaluation Metrics
Three classes of evaluation metrics are considered here. Metrics that evaluate the color consistency with the reference image (indicated with $`^r`$), metrics that evaluate the structural similarity with the source image (indicated with $`^s`$) and metrics that evaluates the overall quality of the output (indicated with $`^o`$).

<details>
  <summary>List of integrated Objective Evaluation Metrics</summary>

### SSIM$`^s_{rgb}`$
**Name**: Structural Similarity Index  
**Description**: The Structural Similarity Index (SSIM) is a metric used to measure the similarity between two images. Unlike traditional metrics like Mean Squared Error (MSE) that focus solely on pixel-wise differences, SSIM considers changes in structural information, luminance, and texture. The index provides a value between -1 and 1, where a value of 1 indicates that the two images being compared are identical in terms of structural content. SSIM is widely used in the field of image processing for quality assessment of compressed or processed images in comparison to reference images.

### PSNR$`^s_{rgb}`$
**Name**: Peak Signal-to-Noise Ratio  

### MSE$`^s_{rgb}`$
**Name**: Mean-Squared Error  

### MS-SSIM$`^s_{rgb}`$
**Name**: Multi-Scale Structural Similarity Index  

### FSIM$`^s_{c,yiq}`$
**Name**: Feature Similarity Index  

### LPIPS$`^s_{rgb}`$
**Name**: Learned Perceptual Image Patch Similarity  

### RMSE$`^s_{rgb}`$
**Name**: Root-Mean-Squared Error  

### GSSIM$`^s_{rgb}`$
**Name**: Gradient-based Structural Similarity Index  

### VSI$`^s_{rgb}`$
**Name**: Visual Saliency-based Index 

### 4-SSIM$`^s_{rgb}`$
**Name**: 4-component Structural Similarity Index

### HI$`^r_{rgb}`$
**Name**: Histogram Intersection 

### Corr$`^r_{rgb}`$
**Name**: Correlation 

### BA$`^r_{rgb}`$
**Name**: Bhattacharyya Distance 

### BRISQUE$`^o_{rgb}`$
**Name**: Blind/Referenceless Image Spatial Quality Evaluator 

### NIMA$`^o_{rgb}`$
**Name**: Neural Image Assessment 

### NIQE$`^o_{rgb}`$
**Name**: Naturalness Image Quality Evaluator 

### CF$`^o_{rgyb}`$
**Name**: Colorfulness 

### 4-EGSSIM$`^s_{rgb}`$
**Name**: 4-component enhanced Gradient-based Structural Similarity Index 

### CSS$`^{sr}_{rgb}`$
**Name**: Color and Structure Similarity 

### CTQM$`^{sro}_{lab}`$
**Name**: Color Transfer Quality Metric 

</details>

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

## References
[^1]: E. Reinhard, M. Ashikhmin, B. Gooch, and P. Shirley, “Color transfer between images,” *IEEE Comput. Graph. Appl.*, vol. 21, p. 34–41, sep 2001.
[^2]: F. Pitie, A. C. Kokaram and R. Dahyot, "N-dimensional probability density function transfer and its application to color transfer," *Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1*, 2005, pp. 1434-1439 Vol. 2, doi: 10.1109/ICCV.2005.166.
[^3]: F. Pitie and A. Kokaram, "The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer," *4th European Conference on Visual Media Production*, 2007, pp. 1-9, doi: 10.1049/cp:20070055.
[^4]: X. Qian, BangFeng Wang and Lei Han, "An efficient fuzzy clustering-based color transfer method," *2010 Seventh International Conference on Fuzzy Systems and Knowledge Discovery*, 2010, pp. 520-523, doi: 10.1109/FSKD.2010.5569560.
[^5]: Gatys, Leon A. and Ecker, Alexander S. and Bethge, Matthias, "A Neural Algorithm of Artistic Style," *arXiv*, 2015, doi: 10.48550/arXiv.1508.06576. 
[^6]: Luan, Fujun and Paris, Sylvain and Shechtman, Eli and Bala, Kavita, "Deep Photo Style Transfer," *arXiv*, 2017, doi: 10.48550/arxiv.1703.07511.  
[^7]: Mairéad Grogan, Rozenn Dahyot, "L2 Divergence for robust colour transfer," *Computer Vision and Image Understanding*, 2019, pp. 39-49 Vol. 181 doi: 10.1016/j.cviu.2019.02.002.  
[^8]: Junyong Lee, Hyeongseok Son, Gunhee Lee, Jonghyeop Lee, Sunghyun Cho, Seungyong Le, "Deep Color Transfer using Histogram Analogy," *The Visual Computer*, 2020, pp. 2129-2143 Vol. 36, doi: 10.1007/s00371-020-01921-6.  
[^9]: Cao, Xu and Wang, Weimin and Nagao, Katashi and Nakamura, Ryosuke, "PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color," *IEEE Computer Graphics and Applications*, doi: 110.1109/WACV45572.2020.9093513.  
[^10]: Ific Goudé, Rémi Cozot, Olivier Le Meur, Kadi Bouatouch. Example‐Based Colour Transfer for 3D Point Clouds. Computer Graphics Forum, Wiley, 2021, 40 (6), pp.428-446. ⟨10.1111/cgf.14388⟩. ⟨hal-03396448⟩  
[^11]: Afifi, Mahmoud and Abuolaim, Abdullah and Hussien, Mostafa and Brubaker, Marcus A. and Brown, Michael S, "CAMS: Color-Aware Multi-Style Transfer," *arXiv*, 2021, doi: 10.48550/ARXIV.2106.13920.  
[^12]: Chang, Youngha and Saito, Suguru and Nakajima, Masayuki, "A Framework for Transfer Colors Based on the Basic Color Categories," *Proceedings Computer Graphics International*, 2003, doi: 10.1109/CGI.2003.1214463.  
