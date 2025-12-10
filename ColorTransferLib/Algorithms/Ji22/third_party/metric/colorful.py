import cv2
import numpy as np
import glob
import os



img_root = ''
imgs = glob.glob(os.path.join(img_root, "*/*"))


def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))

	# compute rg = R - G
	rg = np.absolute(R - G)

	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)

	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))

	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

# f=open("one.csv", "w")
colorful = []
for idx, img_path in enumerate(imgs):
	img = cv2.imread(img_path)
	cf = image_colorfulness(img)
	if cf>1:
		colorful.append(cf)
		print(idx, colorful[-1], np.mean(colorful))
		# f.write("{} {}\n".format(img_path, colorful[-1]))
# f.close()
print("AVG: ", np.mean(colorful))
