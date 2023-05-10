from PIL import Image
import style_loss

input_image_name = './result/1.jpg'
target_image_name = './style/1.jpg'

input = Image.open(input_image_name)
target = Image.open(target_image_name)

loss = style_loss.style_loss(input, target)