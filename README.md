# ColorTransferLib
This color transfer library, called *ColorTransferLib* offers a collection of published color transfer algorithms. The ultimate goal of this project is to create a collection of all developed color transfer methods with standardized API in order to support the research community in the developing and the comparison process of their own algorithms. Currently this library supports color transfer with **images** and **3D point clouds** but will be extended by **triangulated and textured 3D meshes** and **videos**.

## API

TODO

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

## Available Objective Evaluation Metrics
Two classes of evaluation metrics are considered here. Metrics that evaluate the color consistency (CC) and metrics that evaluate the structural similarity (SS).
### (1/X) PSNR (SS)
**Name**: Peak Signal-to-Noise Ratio  
TODO

### (2/X) SSIM (SS)
**Name**: Structural Similarity Index  
TODO

### (3/X) GSSIM
**Name**: Gradient-based Structural Similarity Index  
TODO

## Available Color Transfer Methods:
The following color transfer methods are integrated in the library. Some of them are reimplemented based on the algorithm's description in the the published papers and others are adopted from existing repositories and adpated to fit the API. The original implementation of the latter methods can be found next to the **Source** entry.

### (1/12) GlobalColorTransfer[^1] 
**Title**: Color Transfer between Images  
**Author**: Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley  
**Published in**: IEEE Computer Graphics and Applications  
**Year of Publication**: 2001  
**Link**: https://doi.org/10.1109/38.946629  

**Abstract**: *We use a simple statistical analysis to impose one image's color characteristics on another. We can achieve color correction by choosing an appropriate source image and apply its characteristic to another image.*

<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/189818509-cd7e602a-bad2-464c-9e8b-a58d03fd244a.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/189818674-237bbfba-f706-4c34-a806-6a12d2c2391c.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/189818708-a7b83045-79c9-49e6-8957-f0ad11bab0d9.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>

### (2/12) PdfColorTransfer[^2]
**Title**: N-dimensional probability density function transfer and its application to color transfer  
**Author**: Francois Pitie, Anil C. Kokaram, Rozenn Dahyot  
**Published in**: Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1  
**Year of Publication**: 2005  
**Link**: https://doi.org/10.1109/ICCV.2005.166  
  
**Abstract**: *This article proposes an original method to estimate a continuous transformation that maps one N-dimensional distribution to another. The method is iterative, non-linear, and is shown to converge. Only 1D marginal distribution is used in the estimation process, hence involving low computation costs. As an illustration this mapping is applied to color transfer between two images of different contents. The paper also serves as a central focal point for collecting together the research activity in this area and relating it to the important problem of automated color grading.*

<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/190165339-244bf855-85a1-4900-bdd5-bb691d380ed9.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190165357-52fb673b-3b59-488e-bea1-83b2112add80.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190165368-2308bc46-40fd-4a25-a890-f90358987b11.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>
  
### (3/12) MongeKLColorTransfer[^3]
**Title**: The Linear Monge-Kantorovitch Linear Colour Mapping for Example-Based Colour Transfer.  
**Author**: Francois Pitie, Anil C. Kokaram  
**Published in**: 4th European Conference on Visual Media Production  
**Year of Publication**: 2007  
**Link**: https://doi.org/10.1049/cp:20070055  
  
**Abstract**: *A common task in image editing is to change the colours of a picture to match the desired colour grade of another picture. Finding the correct colour mapping is tricky because it involves numerous interrelated operations, like balancing the colours, mixing the colour channels or adjusting the contrast. Recently, a number of automated tools have been proposed to find an adequate one-to-one colour mapping. The focus in this paper is on finding the best linear colour transformation. Linear transformations have been proposed in the literature but independently. The aim of this paper is thus to establish a common mathematical background to all these methods. Also, this paper proposes a novel transformation, which is derived from the Monge-Kantorovitch theory of mass transportation. The proposed solution is optimal in the sense that it minimises the amount of changes in the picture colours. It favourably compares theoretically and experimentally with other techniques for various images and under various colour spaces.*

<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/191666563-6128f147-1aa4-4cab-9b66-34166ce78915.jpg width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/191666570-08b827bb-2b15-4a83-ab99-ce58728bf4af.jpg width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/191666575-6c6fda6a-b3f0-4243-ae4a-9e989658b729.jpg width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>
  
### (4/12) FuzzyColorTransfer[^4] 
**Title**: An efficient fuzzy clustering-based color transfer method  
**Author**: XiaoYan Qian, BangFeng Wang, Lei Han  
**Published in**: Seventh International Conference on Fuzzy Systems and Knowledge Discovery  
**Year of Publication**: 2010  
**Link**: https://doi.org/10.1109/FSKD.2010.5569560  

**Abstract**: *Each image has its own color content that greatly influences the perception of human observer. Recently, color transfer among different images has been under investigation. In this paper, after a brief review on the few efficient works performed in the field, a novel fuzzy clustering based color transfer method is proposed. The proposed method accomplishes the transformation based on a set of corresponding fuzzy clustering algorithm-selected regions in images along with membership degree factors. Results show the presented algorithm is highly automatically and more effective.*
  
<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/189827426-573945d3-47c5-4a2f-b55d-9dbbaeb75ca0.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/189827443-46de8513-0583-410d-8a2a-612e43654e3f.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/189827453-33fcd07d-4d1a-48a6-86b6-ef0e0ee5f3c4.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>
  
### (5/12) NeuralStyleTransfer[^5]
**Title**: A Neural Algorithm of Artistic Style  
**Author**: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge  
**Published in**: arXiv  
**Year of Publication**: 2015  
**Link**: https://doi.org/10.48550/arXiv.1508.06576  
**Source**: https://github.com/cysmith/neural-style-tf  
  
**Abstract**: *In fine art, especially painting, humans have mastered the skill to create unique visual experiences through composing a complex interplay between the content and style of an image. Thus far the algorithmic basis of this process is unknown and there exists no artificial system with similar capabilities. However, in other key areas of visual perception such as object and face recognition near-human performance was recently demonstrated by a class of biologically inspired vision models called Deep Neural Networks. Here we introduce an artificial system based on a Deep Neural Network that creates artistic images of high perceptual quality. The system uses neural representations to separate and recombine content and style of arbitrary images, providing a neural algorithm for the creation of artistic images. Moreover, in light of the striking similarities between performance-optimised artificial neural networks and biological vision, our work offers a path forward to an algorithmic understanding of how humans create and perceive artistic imagery.*

<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/190636023-ab21948b-3fe5-4cc6-8a99-c0b43ae43730.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190636036-cc5992a8-c842-429b-898b-0ccfdbbed836.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190636044-02fb5c59-c93d-4be0-bad7-07e5dee6824c.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>
  
### (6/12) DeepPhotoStyleTransfer[^6] 
**Title**: Deep Photo Style Transfer  
**Author**: Fujun Luan, Sylvain Paris, Eli Shechtman, Kavita Bala  
**Published in**: ...  
**Year of Publication**: 2017  
**Link**: https://doi.org/10.48550/arXiv.1703.07511  
**Source**: https://github.com/LouieYang/deep-photo-styletransfer-tf  

**Abstract**: *This paper introduces a deep-learning approach to photographic style transfer that handles a large variety of image content while faithfully transferring the reference style. Our approach builds upon the recent work on painterly transfer that separates style from the content of an image by considering different layers of a neural network. However, as is, this approach is not suitable for photorealistic style transfer. Even when both the input and reference images are photographs, the output still exhibits distortions reminiscent of a painting. Our contribution is to constrain the transformation from the input to the output to be locally affine in colorspace, and to express this constraint as a custom fully differentiable energy term. We show that this approach successfully suppresses distortion and yields satisfying photorealistic style transfers in a broad variety of scenarios, including transfer of the time of day, weather, season, and artistic edits.*

<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/189828362-7e992cce-a943-4d5b-8b26-d54514fe7bf8.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/189828374-4ce9fa08-1bad-4252-ac4a-ea0496da3039.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/189828389-a8939c13-2797-4bc0-8c16-fd820bb91119.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>
  
### (7/12) TpsColorTransfer[^7]
**Title**: L2 Divergence for robust colour transfer  
**Author**: Mairéad Grogan, Rozenn Dahyot  
**Published in**: Computer Vision and Image Understanding  
**Year of Publication**: 2019  
**Link**: https://doi.org/10.1016/j.cviu.2019.02.002  
**Source**: https://github.com/groganma/gmm-colour-transfer  
  
**Abstract**: *Optimal Transport (OT) is a very popular framework for performing colour transfer in images and videos. We have proposed an alternative framework where the cost function used for inferring a parametric transfer function is defined as the robust L 2 divergence between two probability density functions (Grogan and Dahyot, 2015). In this paper, we show that our approach combines many advantages of state of the art techniques and outperforms many recent algorithms as measured quantitatively with standard quality metrics, and qualitatively using perceptual studies (Grogan and Dahyot, 2017). Mathematically, our formulation is presented in contrast to the OT cost function that shares similarities with our cost function. Our formulation, however, is more flexible as it allows colour correspondences that may be available to be taken into account and performs well despite potential occurrences of correspondence outlier pairs. Our algorithm is shown to be fast, robust and it easily allows for user interaction providing freedom for artists to fine tune the recoloured images and videos (Grogan et al., 2017).*
  
<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/189824405-7beedb7f-7463-4fa2-a7b9-3c5b766919f0.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/189824419-9988d961-619d-4cce-92f0-d87ad53f6e3b.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/189824430-f8e11d82-38f9-4283-a602-5e250c2083b7.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>
  
### (8/12) HistogramAnalogy[^8]
**Title**: Deep Color Transfer using Histogram Analogy  
**Author**: Junyong Lee, Hyeongseok Son, Gunhee Lee, Jonghyeop Lee, Sunghyun Cho, Seungyong Lee  
**Published in**: The Visual Computer: International Journal of Computer Graphics, Volume 36, Issue 10-12Oct 2020  
**Year of Publication**: 2020  
**Link**: https://doi.org/10.1007/s00371-020-01921-6  
**Source**: https://github.com/codeslake/Color_Transfer_Histogram_Analogy  
  
**Abstract**: *We propose a novel approach to transferring the color of a reference image to a given source image. Although there can be diverse pairs of source and reference images in terms of content and composition similarity, previous methods are not capable of covering the whole diversity. To resolve this limitation, we propose a deep neural network that leverages color histogram analogy for color transfer. A histogram contains essential color information of an image, and our network utilizes the analogy between the source and reference histograms to modulate the color of the source image with abstract color features of the reference image. In our approach, histogram analogy is exploited basically among the whole images, but it can also be applied to semantically corresponding regions in the case that the source and reference images have similar contents with different compositions. Experimental results show that our approach effectively transfers the reference colors to the source images in a variety of settings. We also demonstrate a few applications of our approach, such as palette-based recolorization, color enhancement, and color editing.*
  
<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/190618388-b75f76f3-afff-47e6-bc1d-d5d73aeecf93.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190618403-8d5abdad-3c58-4b1e-91dc-8efde94455a3.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190618410-ec7a6912-8533-4ce8-83bd-8d3d8fd9cabc.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>

### (9/12) PSNetStyleTransfer[^9]  
**Title**: PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color  
**Author**: Cao, Xu and Wang, Weimin and Nagao, Katashi and Nakamura, Ryosuke  
**Published in**: IEEE Computer Graphics and Applications  
**Year of Publication**: 2020  
**Link**: https://doi.org/10.1109/WACV45572.2020.9093513  
**Source**: https://github.com/hoshino042/psnet  
  
**Abstract**: *We propose a neural style transfer method for colored point clouds which allows stylizing the geometry and/or color property of a point cloud from another. The stylization is achieved by manipulating the content representations and Gram-based style representations extracted from a pretrained PointNet-based classification network for colored point clouds. As Gram-based style representation is invariant to the number or the order of points, the style can also be an image in the case of stylizing the color property of a point cloud by merely treating the image as a set of pixels. Experimental results and analysis demonstrate the capability of the proposed method for stylizing a point cloud either from another point cloud or an image.*
  
<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/191665829-77bc1742-65f2-4d26-a149-ecebbe868a84.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/191665840-1eb736e2-acf9-4232-bf76-604684a8ae5e.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/191665848-284d8f55-bfcc-46c5-878f-be11ca134b5e.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>

### (10/12) Eb3dColorTransfer[^10]
**Title**: Example-Based Colour Transfer for 3D Point Clouds  
**Author**: Ific Goudé, Rémi Cozot, Olivier Le Meur, Kadi Bouatouch  
**Published in**: ...  
**Year of Publication**: 2021  
**Link**: https://doi.org/10.1111/cgf.14388  
  
**Abstract**: *Example-based colour transfer between images, which has raised a lot of interest in the past decades, consists of transferring the colour of an image to another one. Many methods based on colour distributions have been proposed, and more recently, the efficiency of neural networks has been demonstrated again for colour transfer problems. In this paper, we propose a new pipeline with methods adapted from the image domain to automatically transfer the colour from a target point cloud to an input point cloud. These colour transfer methods are based on colour distributions and account for the geometry of the point clouds to produce a coherent result. The proposed methods rely on simple statistical analysis, are effective, and succeed in transferring the colour style from one point cloud to another. The qualitative results of the colour transfers are evaluated and compared with existing methods.*

<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/190633991-21a703b0-1dcc-47c5-9801-bdb929551623.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190633994-8e4498dd-73c1-4e08-a4e7-c0773bd13dc4.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190634002-a18854c5-393f-44f6-8b58-72db7b6286c5.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>

### (11/12) CamsTransfer[^11] 
**Title**: CAMS: Color-Aware Multi-Style Transfer  
**Author**: Mahmoud Afifi, Abdullah Abuolaim, Mostafa Hussien, Marcus A. Brubaker, Michael S. Brown  
**Published in**: arXiv  
**Year of Publication**: 2021  
**Link**: https://doi.org/10.48550/arXiv.2106.13920  
**Source**: https://github.com/mahmoudnafifi/color-aware-style-transfer  

**Abstract**: *Image style transfer aims to manipulate the appearance of a source image, or "content" image, to share similar texture and colors of a target "style" image. Ideally, the style transfer manipulation should also preserve the semantic content of the source image. A commonly used approach to assist in transferring styles is based on Gram matrix optimization. One problem of Gram matrix-based optimization is that it does not consider the correlation between colors and their styles. Specifically, certain textures or structures should be associated with specific colors. This is particularly challenging when the target style image exhibits multiple style types. In this work, we propose a color-aware multi-style transfer method that generates aesthetically pleasing results while preserving the style-color correlation between style and generated images. We achieve this desired outcome by introducing a simple but efficient modification to classic Gram matrix-based style transfer optimization. A nice feature of our method is that it enables the users to manually select the color associations between the target style and content image for more transfer flexibility. We validated our method with several qualitative comparisons, including a user study conducted with 30 participants. In comparison with prior work, our method is simple, easy to implement, and achieves visually appealing results when targeting images that have multiple styles.*

<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/190634689-5073d7c4-4854-4911-8486-d120decc2588.jpg width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190634696-2d40d3cd-f667-4fb8-b958-cdeaa114372f.jpg width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190634701-0f1ea113-5368-4b84-a029-d72462ae1b7a.jpg width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>
  
### (12/12) GmmEmColorTransfer[^12]
**Title**: Example-based Color Transfer with Gaussian Mixture Modeling  
**Author**: Chunzhi Gu, Xuequan Lu,and Chao Zhang  
**Published in**: Journal of Computer-Aided Design and Computer Graphics  
**Year of Publication**: 2022  
**Link**: https://doi.org/10.1016/j.patcog.2022.108716  
  
**Abstract**: *Color transfer, which plays a key role in image editing, has attracted noticeable attention recently. It has remained a challenge to date due to various issues such as time-consuming manual adjustments and prior segmentation issues. In this paper, we propose to model color transfer under a probability framework and cast it as a parameter estimation problem. In particular, we relate the transferred image with the example image under the Gaussian Mixture Model (GMM) and regard the transferred image color as the GMM centroids. We employ the Expectation-Maximization (EM) algorithm (E-step and M-step) for optimization. To better preserve gradient information, we introduce a Laplacian based regularization term to the objective function at the M-step which is solved by deriving a gradient descent algorithm. Given the input of a source image and an example image, our method is able to generate multiple color transfer results with increasing EM iterations. Extensive experiments show that our approach generally outperforms other competitive color transfer methods, both visually and quantitatively.*

<table>
  <tr>
    <td><img src=https://user-images.githubusercontent.com/15614886/190166732-da75ba72-2096-4f66-bc2d-e9c3ce439b7a.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190166723-46df5774-b434-4e7a-a4ec-5def9ae36794.png width=512px/></td>
    <td><img src=https://user-images.githubusercontent.com/15614886/190166740-bd8f84b0-e995-4a75-b986-55c55b0f9554.png width=512px/></td>
  </tr>
  <tr>
    <td>Source</td>
    <td>Reference</td>
    <td>Result</td>
  </tr>
<table>
  
## References
[^1]: E. Reinhard, M. Ashikhmin, B. Gooch, and P. Shirley, “Color transfer between images,” *IEEE Comput. Graph. Appl.*, vol. 21, p. 34–41, sep 2001.
[^2]: F. Pitie, A. C. Kokaram and R. Dahyot, "N-dimensional probability density function transfer and its application to color transfer," *Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1*, 2005, pp. 1434-1439 Vol. 2, doi: 10.1109/ICCV.2005.166.
[^3]: F. Pitie and A. Kokaram, "The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer," *4th European Conference on Visual Media Production*, 2007, pp. 1-9, doi: 10.1049/cp:20070055.
[^4]: X. Qian, BangFeng Wang and Lei Han, "An efficient fuzzy clustering-based color transfer method," *2010 Seventh International Conference on Fuzzy Systems and Knowledge Discovery*, 2010, pp. 520-523, doi: 10.1109/FSKD.2010.5569560.
[^5]: Gatys, Leon A. and Ecker, Alexander S. and Bethge, Matthias, "A Neural Algorithm of Artistic Style," *arXiv*, 2015, doi: 10.48550/arXiv.1508.06576. 
[^6]: Luan, Fujun and Paris, Sylvain and Shechtman, Eli and Bala, Kavita, "Deep Photo Style Transfer," *arXiv*, 2017, doi: 10.48550/arxiv.1703.07511.  
[^7]: Mairéad Grogan, Rozenn Dahyot, "L2 Divergence for robust colour transfer," *Computer Vision and Image Understanding*, 2019, pp. 39-49 Vol. 181 doi: 10.1016/j.cviu.2019.02.002.  
[^8]: Junyong Lee, Hyeongseok Son, Gunhee Lee, Jonghyeop Lee, Sunghyun Cho, Seungyong Le, "Deep Color Transfer using Histogram Analogy," *The Visual Computer*, 2020, pp. 2129-2143 Vol. 36, doi: 10.1007/s00371-020-01921-6.  
[^9]: JCao, Xu and Wang, Weimin and Nagao, Katashi and Nakamura, Ryosuke, "PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color," *IEEE Computer Graphics and Applications*, doi: 110.1109/WACV45572.2020.9093513.  
[^10]: Ific Goudé, Rémi Cozot, Olivier Le Meur, Kadi Bouatouch. Example‐Based Colour Transfer for 3D Point Clouds. Computer Graphics Forum, Wiley, 2021, 40 (6), pp.428-446. ⟨10.1111/cgf.14388⟩. ⟨hal-03396448⟩  
[^11]: Afifi, Mahmoud and Abuolaim, Abdullah and Hussien, Mostafa and Brubaker, Marcus A. and Brown, Michael S, "CAMS: Color-Aware Multi-Style Transfer," *arXiv*, 2021, doi: 10.48550/ARXIV.2106.13920.  
[^12]: Gu, Chunzhi and Lu, Xuequan and Zhang, Chao, "Example-Based Color Transfer with Gaussian Mixture Modeling," *Pattern Recogn.*, 2022, Vol. 129  doi: 10.1016/j.patcog.2022.108716.  
