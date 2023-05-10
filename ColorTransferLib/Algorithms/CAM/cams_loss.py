"""
CAMS loss implementation.

Reference:
M. Afifi, A. Abuolaim, M. Korashy,  M. A. Brubaker, and M. S. Brown. Color-Aware Style Transfer. arXiv preprint 2021.


# Libraries
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import copy
import cv2
import math

"""Settings"""
SMOOTH = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
SIGMA = 0.25
PALETTE_SIZE = 5
ADD_BLACK_WHITE = False
COLOR_DISTANCE = 'chroma_L2'  # Options: 'chroma_L2', 'L2'
STYLE_FEATURE_DISTANCE = 'L2'  # Options: 'L2', 'COSINE'
IMAGE_SIZE = 384
# desired size of the output image
imsize = IMAGE_SIZE if torch.cuda.is_available() else 128
# desired depth layers to compute style/content losses :
color_aware_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


"""Loading VGG model"""
cnn = models.vgg19(pretrained=True).features.to(DEVICE).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)


"""If SMOOTH is true, create a Gaussian blur kernel."""
if SMOOTH:
  """# Gaussian blur kernel"""

  # Set these to whatever you want for your gaussian filter
  kernel_size = 15
  sigma = 5

  # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
  x_cord = torch.arange(kernel_size)
  x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1)

  mean = (kernel_size - 1) / 2.
  variance = sigma ** 2.

  # Calculate the 2-dimensional gaussian kernel which is
  # the product of two gaussian distributions for two different
  # variables (in this case called x and y)
  gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                    torch.exp(
                      -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                      (2 * variance))
  # Make sure sum of values in gaussian kernel equals 1.
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

  # Reshape to 2d depthwise convolutional weight
  gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
  gaussian_kernel = gaussian_kernel.to(device=DEVICE)


"""Image Loader"""
loader = transforms.Compose([
  transforms.Resize((imsize, imsize)),  # scale imported image
  transforms.ToTensor()])  # transform it into a torch tensor

"""Color palette
Ref: https://github.com/tody411/PaletteSelection
"""

"""Helper functions"""
## Convert image into float32 type.
def to32F(img):
  if img.dtype == np.float32:
    return img
  return (1.0 / 255.0) * np.float32(img)


## RGB channels of the image.
def rgb(img):
  if len(img.shape) == 2:
    h, w = img.shape
    img_rgb = np.zeros((h, w, 3), dtype=img.dtype)
    for ci in range(3):
      img_rgb[:, :, ci] = img
    return img_rgb

  h, w, cs = img.shape
  if cs == 3:
    return img

  img_rgb = np.zeros((h, w, 3), dtype=img.dtype)

  cs = min(3, cs)

  for ci in range(cs):
    img_rgb[:, :, ci] = img[:, :, ci]
  return img_rgb


## RGB to Lab.
def rgb2Lab(img):
  img_rgb = rgb(img)
  Lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
  return Lab


## Lab to RGB.
def Lab2rgb(img):
  rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
  return rgb


def _isGray(image):
  return len(image.shape) == 2


## True if x is a vector.
def isVector(x):
  return x.size == x.shape[0]


## True if x is a matrix.
def isMatrix(x):
  return not isVector(x)


## Norm of vectors (n x m matrix).
def normVectors(x):
  return np.sqrt(l2NormVectors(x))


## L2 norm of vectors (n x m matrix).
#  n x 1 vector: call np.square.
#  n x m vectors: call np.einsum.
def l2NormVectors(x):
  if isVector(x):
    return np.square(x)
  else:
    return np.einsum('...i,...i', x, x)


def colorCoordinates(color_ids, num_bins, color_range):
  color_ids = np.array(color_ids).T
  c_min, c_max = color_range
  color_coordinates = c_min + (
      color_ids * (c_max - c_min)) / float(num_bins - 1.0)
  return color_coordinates


def colorDensities(hist_bins):
  hist_positive = hist_bins > 0.0
  color_densities = np.float32(hist_bins[hist_positive])

  density_max = np.max(color_densities)
  color_densities = color_densities / density_max

  return color_densities


def rgbColors(hist_bins, color_bins):
  hist_positive = hist_bins > 0.0

  colors = color_bins[hist_positive, :]
  colors = np.clip(colors, 0.0, 1.0)
  return colors


def clipLowDensity(hist_bins, color_bins, alpha):
  density_mean = np.mean(hist_bins)
  low_density = hist_bins < density_mean * alpha
  hist_bins[low_density] = 0.0

  for ci in range(3):
    color_bins[low_density, ci] = 0.0


def densitySizes(color_densities, density_size_range):
  density_size_min, density_size_max = density_size_range
  density_size_factor = density_size_max / density_size_min
  density_sizes = density_size_min * np.power(
    density_size_factor, color_densities)
  return density_sizes


def range2ticks(tick_range, decimals=1):
  ticks = np.around(tick_range, decimals=decimals)
  ticks[ticks > 10] = np.rint(ticks[ticks > 10])
  return ticks


def range2lims(tick_range):
  unit = 0.1 * (tick_range[:, 1] - tick_range[:, 0])
  lim = np.array(tick_range)
  lim[:, 0] += -unit
  lim[:, 1] += unit

  return lim


"""Color pixels class"""
class ColorPixels:

  def __init__(self, image, num_pixels=1000):
    self._image = to32F(image)
    self._num_pixels = num_pixels
    self._rgb_pixels = None
    self._Lab = None
    self._hsv = None

  ## RGB pixels.
  def rgb(self):
    if self._rgb_pixels is None:
      self._rgb_pixels = self.pixels("rgb")
    return self._rgb_pixels

  ## Lab pixels.
  def Lab(self):
    if self._Lab is None:
      self._Lab = self.pixels("Lab")
    return self._Lab

  ## Pixels of the given color space.
  def pixels(self, color_space="rgb"):
    image = np.array(self._image)

    if color_space == "Lab":
      image = rgb2Lab(self._image)

    return self._image2pixels(image)

  def _image2pixels(self, image):
    if _isGray(image):
      h, w = image.shape
      step = int(h * w / self._num_pixels)
      return image.reshape((h * w))[::step]

    h, w, cs = image.shape
    step = int(h * w / self._num_pixels)
    return image.reshape((-1, cs))[::step]


"""3D color histograms"""
class Hist3D:
  def __init__(self, image,
               num_bins=16, alpha=0.1, color_space='rgb'):
    self._computeTargetPixels(image, color_space)
    self._num_bins = num_bins
    self._alpha = alpha
    self._color_space = color_space
    self._computeColorRange()
    self._computeHistogram()

  def colorSpace(self):
    return self._color_space

  def colorIDs(self):
    color_ids = np.where(self._histPositive())
    return color_ids

  def colorCoordinates(self):
    color_ids = self.colorIDs()
    num_bins = self._num_bins
    color_range = self._color_range
    return colorCoordinates(color_ids, num_bins, color_range)

  def colorDensities(self):
    return colorDensities(self._hist_bins)

  def rgbColors(self):
    return rgbColors(self._hist_bins, self._color_bins)

  def colorRange(self):
    return self._color_range

  def _computeTargetPixels(self, image, color_space):
    color_pixels = ColorPixels(image)
    self._pixels = color_pixels.pixels(color_space)
    self._rgb_pixels = color_pixels.rgb()

  def _computeColorRange(self):
    pixels = self._pixels
    cs = pixels.shape[1]

    c_min = np.zeros(cs)
    c_max = np.zeros(cs)
    for ci in range(cs):
      c_min[ci] = np.min(pixels[:, ci])
      c_max[ci] = np.max(pixels[:, ci])

    self._color_range = [c_min, c_max]

  def _computeHistogram(self):
    pixels = self._pixels
    num_bins = self._num_bins
    c_min, c_max = self._color_range

    hist_bins = np.zeros((num_bins, num_bins, num_bins), dtype=np.float32)
    color_bins = np.zeros((num_bins, num_bins, num_bins, 3),
                          dtype=np.float32)

    color_ids = (num_bins - 1) * (pixels - c_min) / (c_max - c_min)
    color_ids = np.int32(color_ids)

    for pi, color_id in enumerate(color_ids):
      hist_bins[color_id[0], color_id[1], color_id[2]] += 1
      color_bins[color_id[0], color_id[1],
                 color_id[2]] += self._rgb_pixels[pi]

    self._hist_bins = hist_bins
    hist_positive = self._hist_bins > 0.0

    for ci in range(3):
      color_bins[hist_positive, ci] /= self._hist_bins[hist_positive]

    self._color_bins = color_bins

    self._clipLowDensity()

  def _clipLowDensity(self):
    clipLowDensity(self._hist_bins, self._color_bins, self._alpha)

  def _histPositive(self):
    return self._hist_bins > 0.0


"""Auto palette selection"""
class PaletteSelection:
  def __init__(self, color_coordinates, color_densities, rgb_colors,
               num_colors=7, sigma=70.0):
    self._color_coordinates = color_coordinates
    self._color_densities = color_densities
    self._rgb_colors = rgb_colors
    self._num_colors = num_colors
    self._sigma = sigma
    self._palette_coordinates = []
    self._palette_colors = []
    self._computeDarkBrightColors()
    self._computeInitialWeight()
    self._compute()

  def paletteCoordinates(self):
    return self._palette_coordinates

  def paletteColors(self):
    return self._palette_colors

  def _compute(self):
    for i in range(self._num_colors):
      palette_coordinate = self._updatePalette()
      self._updateWeight(palette_coordinate)

  def _computeDarkBrightColors(self):
    rgb_colors = self._rgb_colors

    intensities = normVectors(rgb_colors)
    c_dark = self._color_coordinates[np.argmin(intensities)]
    c_bright = self._color_coordinates[np.argmax(intensities)]
    self._dark_bright = [c_dark, c_bright]

  def _computeInitialWeight(self):
    self._color_weights = np.array(self._color_densities)
    self._updateWeight(self._dark_bright[0])
    self._updateWeight(self._dark_bright[1])

  def _updatePalette(self):
    color_id = np.argmax(self._color_weights)
    palette_coordinate = self._color_coordinates[color_id]
    self._palette_coordinates.append(palette_coordinate)

    palette_color = self._rgb_colors[color_id]
    self._palette_colors.append(palette_color)
    return palette_coordinate

  def _updateWeight(self, palette_coordinate):
    dists = normVectors(self._color_coordinates - palette_coordinate)
    factors = 1.0 - np.exp(- dists ** 2 / (self._sigma ** 2))
    self._color_weights = factors * self._color_weights


"""Mask generation"""
class CreateMask(nn.Module):
  def __init__(self, insz=imsize, color_palette=None, sigma=SIGMA,
               smooth=SMOOTH, distance=COLOR_DISTANCE):
    """ Computes masks of the image based on a given color palette
    Args:
      insz: maximum size of the input image; if it is larger than this size, the
        image will be resized (scalar). Default value is imsize (i.e.,
        imsize x imsize pixels).
      color_palette: kx3 tensor of color palette
      sigma: this is the sigma parameter of the kernel function.
        The default value is 0.02.
      smooth: boolean flag to apply a Gaussian blur after creating the mask.
      distance: it can be one of the following options: 'chroma_L2' or 'L2'

    Methods:
      forward: accepts input image and returns its masks based on the input
      color palette
    """
    super(CreateMask, self).__init__()
    self.color_palette = color_palette
    self.insz = insz
    self.device = DEVICE
    self.sigma = sigma
    self.distance = distance
    self.smooth = smooth

  def forward(self, x):
    if self.color_palette is None:
      raise NameError('No color palette is given')
    x = torch.clamp(x, 0, 1)
    I = F.interpolate(x, size=(self.insz, self.insz),
                      mode='bilinear', align_corners=False)
    masks = torch.zeros(1, self.color_palette.shape[0], self.insz, self.insz,
                        device=DEVICE)
    if I.shape[1] > 3:
      I = I[:, :3, :, :]
    if self.distance == 'chroma_L2':
      I = I / (torch.unsqueeze(torch.sum(I, dim=1), dim=1) + EPS)

    for c in range(self.color_palette.shape[0]):
      color = self.color_palette[c, :].view(1, 3, 1, 1)
      if self.distance == 'chroma_L2':
        color = color / (torch.unsqueeze(torch.sum(color, dim=1), dim=1) +
                         EPS)
      dist = torch.sqrt(torch.sum((I - color) ** 2, dim=1))
      weight = torch.exp(-1 * (dist / self.sigma) ** 2)
      if self.smooth:
        weight = nn.functional.conv2d(torch.unsqueeze(weight, dim=0),
                                      gaussian_kernel,
                                      bias=None, stride=1, padding=7)
      masks[0, c, :, :] = weight
    return masks


"""Loss Functions"""

"""Masked Gram matrix"""
def masked_gram_matrix(input, masks):
  k = masks.shape[1]
  a, b, c, d = input.size()
  masks = F.interpolate(masks, size=(c, d), mode='bilinear',
                        align_corners=False)
  G = torch.zeros(k, a * b * a * b, device=DEVICE)
  features = input.view(a * b, c * d)
  for i in range(k):
    mask_values = masks[:, i, :, :].view(a, c * d)
    mask_values = (mask_values - torch.min(mask_values)) / (
        torch.max(mask_values) - torch.min(mask_values))
    num_elements = torch.sum(mask_values)
    # compute the gram product
    weighted_features = features * mask_values
    g = torch.mm(weighted_features, weighted_features.t())
    G[i, :] = g.div(num_elements).view(1, a * b * a * b)
  return G / k

"""Cosine similarity"""
def cosine_similarity(x, y):
  x = x.view(1, -1)
  y = y.view(1, -1)
  return 1 - (torch.sum(x * y) / (x.norm(2) * y.norm(2) + EPS))


"""Color-aware loss"""
class ColorAwareLoss(nn.Module):
  def __init__(self, target_feature, target_masks):
    super(ColorAwareLoss, self).__init__()
    self.target = masked_gram_matrix(target_feature, target_masks).detach()
    self.input_masks = target_masks

  def set_input_masks(self, input_masks):
    self.input_masks = input_masks

  def forward(self, input):
    G = masked_gram_matrix(input, self.input_masks)
    if STYLE_FEATURE_DISTANCE == 'L2':
      self.loss = F.mse_loss(G, self.target)
    elif STYLE_FEATURE_DISTANCE == 'COSINE':
      self.loss = cosine_similarity(G, self.target)
    else:
      raise NotImplementedError

    return input


"""Image loader"""
def preprocessing(image, K=16):
  # compute color palette
  img_array = np.array(image)
  img_array = img_array[:, :, :3]
  # 16 bins, Lab color space
  hist3D = Hist3D(img_array, num_bins=16, color_space='Lab')
  color_coordinates = hist3D.colorCoordinates()
  color_densities = hist3D.colorDensities()
  rgb_colors = hist3D.rgbColors()
  palette_selection = PaletteSelection(color_coordinates, color_densities,
                                       rgb_colors, num_colors=K, sigma=70.0)
  colors = palette_selection._palette_colors
  # fake batch dimension required to fit network's input dimensions
  image = loader(image).unsqueeze(0)
  image = image[:, :3, :, :]
  return image.to(DEVICE, torch.float), torch.tensor(colors).to(DEVICE,
                                                                torch.float)


"""Normalization"""
class Normalization(nn.Module):
  def __init__(self, mean, std):
    super(Normalization, self).__init__()
    # .view the mean and std to make them [C x 1 x 1] so that they can
    # directly work with image Tensor of shape [B x C x H x W].
    # B is batch size. C is number of channels. H is height and W is width.
    self.mean = torch.tensor(mean).view(-1, 1, 1)
    self.std = torch.tensor(std).view(-1, 1, 1)

  def forward(self, img):
    # normalize img
    return (img - self.mean) / self.std


"""Get style model and CAMS loss"""
def get_style_model_and_cams_loss(cnn, normalization_mean, normalization_std,
                                  target_img, target_img_masks,
                                  color_aware_layers=color_aware_layers_default):
  cnn = copy.deepcopy(cnn)

  # normalization module
  normalization = Normalization(normalization_mean,
                                normalization_std).to(DEVICE)

  color_aware_style_losses = []

  # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
  # to put in modules that are supposed to be activated sequentially
  model = nn.Sequential(normalization)

  i = 0  # increment every time we see a conv

  for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
      i += 1
      name = 'conv_{}'.format(i)
    elif isinstance(layer, nn.ReLU):
      name = 'relu_{}'.format(i)
      # The in-place version doesn't play very nicely with the ContentLoss
      # and StyleLoss we insert below. So we replace with out-of-place
      # ones here.
      layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
      name = 'pool_{}'.format(i)
    elif isinstance(layer, nn.BatchNorm2d):
      name = 'bn_{}'.format(i)
    else:
      raise RuntimeError('Unrecognized layer: {}'.format(
        layer.__class__.__name__))

    model.add_module(name, layer)

    if name in color_aware_layers:
      # add style loss:
      target_feature = model(target_img).detach()
      color_aware_loss = ColorAwareLoss(target_feature, target_img_masks)
      model.add_module("color_aware_loss_{}".format(i), color_aware_loss)
      color_aware_style_losses.append(color_aware_loss)

  for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ColorAwareLoss):
      break
  model = model[:(i + 1)]

  return model, color_aware_style_losses


def cams_loss(input, target):
  input, input_palette = preprocessing(input)

  target, target_palette = preprocessing(target)

  assert input.size() == target.size(), \
    "input and target images should have the same size"


  """ Mask generation"""

  final_palette = torch.cat([input_palette, target_palette], dim=0)
  if ADD_BLACK_WHITE:
    black_white = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                               dtype=torch.float32, device=DEVICE)
    final_palette = torch.cat([final_palette, black_white], dim=0)

  final_palette = torch.unique(final_palette, dim=0)

  mask_generator = CreateMask(color_palette=final_palette)

  input_masks = mask_generator(input)

  target_masks = mask_generator(target)

  model, cams_loss_layers = get_style_model_and_cams_loss(
    cnn, cnn_normalization_mean, cnn_normalization_std, target, target_masks)

  for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ColorAwareLoss):
      model[i].set_input_masks(input_masks)

  model(input)

  color_aware_score = 0

  for cal in cams_loss_layers:
    color_aware_score += cal.loss

  cams_loss_value = color_aware_score.item()
  print(f'Color-aware loss: {cams_loss_value}')

  return cams_loss_value
