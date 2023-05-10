from PIL import Image
import content_loss

input_image_name = './result/1.jpg'
target_image_name = './content/1.jpg'

input = Image.open(input_image_name)
target = Image.open(target_image_name)

loss = content_loss.content_loss(input, target)