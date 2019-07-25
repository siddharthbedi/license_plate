from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import os, sys


def ocr(path):
	img = Image.open(path)
	#img = img.convert('RGBA')
	pix = img.load()
	for y in range(img.size[1]):
	    for x in range(img.size[0]):
	        if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
	            pix[x, y] = (0, 0, 0, 255)
	        else:
	            pix[x, y] = (255, 255, 255, 255)
	img.save('./output/temp.jpg')
	text = pytesseract.image_to_string(Image.open('./output/temp.jpg'))
	os.remove('./output/temp.jpg')
	print(text)


path = "./output/license"
dirs = os.listdir( path )
for item in dirs:

  	im = path+'/'+item
  	ocr(im)

