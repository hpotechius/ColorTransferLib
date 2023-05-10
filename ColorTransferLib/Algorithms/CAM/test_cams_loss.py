from PIL import Image
import cams_loss


input_image_name = './result/1.jpg'
target_image_name = './style/1.jpg'

input = Image.open(input_image_name)
target = Image.open(target_image_name)

loss = cams_loss.cams_loss(input, target)