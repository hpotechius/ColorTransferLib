from PIL import Image
import style_loss
import content_loss
import cams_loss
import os
import numpy as np

dir_results = './result_dir/'
dir_styles = './style_dir/'
dir_content = './content_dir/'

res_filenames = [dir_results + f for f in os.listdir(dir_results) if f.endswith(
  ('.JPG', '.jpg', '.png', '.PNG'))]

style_filenames = [dir_styles + f for f in os.listdir(dir_styles) if f.endswith(
  ('.JPG', '.jpg', '.png', '.PNG'))]

content_filenames = [dir_content + f for f in os.listdir(
  dir_content) if f.endswith(('.JPG', '.jpg', '.png', '.PNG'))]

running_style_loss = []
running_content_loss = []
running_cams_loss = []

for result_file, style_file, content_file in zip(
    res_filenames, style_filenames, content_filenames):
  result = Image.open(result_file)
  style = Image.open(style_file)
  content = Image.open(content_file)

  s_loss = style_loss.style_loss(result, style)
  c_loss = content_loss.content_loss(result, content)
  ca_loss = cams_loss.cams_loss(result, style)
  running_style_loss.append(s_loss)
  running_content_loss.append(c_loss)
  running_cams_loss.append(ca_loss)

print('-----------------------------')

print(f'Mean color-aware loss {np.mean(running_cams_loss)}, '
      f'Mean style loss {np.mean(running_style_loss)}, '
      f'Mean content loss {np.mean(running_content_loss)}')
