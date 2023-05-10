"""
Content loss implementation.

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


"""Settings"""
SMOOTH = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
STYLE_FEATURE_DISTANCE = 'L2'  # Options: 'L2', 'COSINE'
IMAGE_SIZE = 384
# desired size of the output image
imsize = IMAGE_SIZE if torch.cuda.is_available() else 128
# desired depth layers to compute style/content losses :
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


"""Loading VGG model"""
cnn = models.vgg19(pretrained=True).features.to(DEVICE).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)


"""Image Loader"""
loader = transforms.Compose([
  transforms.Resize((imsize, imsize)),  # scale imported image
  transforms.ToTensor()])  # transform it into a torch tensor


"""Gram matrix"""
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
  def __init__(self, target):
    super(StyleLoss, self).__init__()
    self.target = gram_matrix(target).detach()

  def forward(self, input):
    G = gram_matrix(input)
    if STYLE_FEATURE_DISTANCE == 'L2':
      self.loss = F.mse_loss(G, self.target)
    elif STYLE_FEATURE_DISTANCE == 'COSINE':
      self.loss = cosine_similarity(G, self.target)
    else:
      raise NotImplementedError

    return input

"""Cosine similarity"""
def cosine_similarity(x, y):
  x = x.view(1, -1)
  y = y.view(1, -1)
  return 1 - (torch.sum(x * y) / (x.norm(2) * y.norm(2) + EPS))



"""Image loader"""
def preprocessing(image):
  # compute color palette
  image = loader(image).unsqueeze(0)
  image = image[:, :3, :, :]
  return image.to(DEVICE, torch.float)


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


"""Get style model and style loss"""
def get_style_model_and_style_loss(cnn, normalization_mean,
                                     normalization_std, target_image,
                                     style_layers=style_layers_default):
  cnn = copy.deepcopy(cnn)

  # normalization module
  normalization = Normalization(normalization_mean,
                                normalization_std).to(DEVICE)

  style_losses = []

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

    if name in style_layers:
      # add content loss:
      target = model(target_image).detach()
      style_loss = StyleLoss(target)
      model.add_module("style_loss_{}".format(i), style_loss)
      style_losses.append(style_loss)

  for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], StyleLoss):
      break
  model = model[:(i + 1)]

  return model, style_losses


def style_loss(input, target):
  input = preprocessing(input)

  target = preprocessing(target)

  assert input.size() == input.size(), \
    "input and target images should have the same size"


  """ Mask generation"""



  model, style_loss_layers = get_style_model_and_style_loss(
    cnn, cnn_normalization_mean, cnn_normalization_std, target)


  model(input)

  style_loss = 0

  for cl in style_loss_layers:
    style_loss += cl.loss

  style_loss_value = style_loss.item()
  print(f'Style loss: {style_loss_value}')

  return style_loss_value
