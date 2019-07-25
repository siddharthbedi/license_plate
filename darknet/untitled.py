from PIL import Image, ImageEnhance
import pytesseract
from car_detect import *


def ocr(path):
  img = Image.open(path)
  #img = img.convert('RGBA')
  '''pix = img.load()
  for y in range(img.size[1]):
      for x in range(img.size[0]):
          if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
              pix[x, y] = (0, 0, 0, 255)
          else:
              pix[x, y] = (255, 255, 255, 255)'''

  #img = ImageEnhance.Brightness(img).enhance(2.0)
  img = ImageEnhance.Contrast(img).enhance(2.0)
  #img = ImageEnhance.Contrast(img).enhance(4.0)
  #img = ImageEnhance.Brightness(img).enhance(0.2)
  #img = ImageEnhance.Contrast(img).enhance(16.0)
  #img.save("./output/temp.jpg")

  #cont = ImageEnhance.Contrast(img)
  #cont.enhance(2)
  #cont.show()
  #cont.save('./output/temp.jpg')
  text = pytesseract.image_to_string(img)
  #t = tesseract_ocr.text_for_filename('./output/temp.jpg')
 # t = tesserocr.image_to_text(Image.open('./output/temp.jpg'))
  #os.remove('./output/temp.jpg')
  #print(text)
  text = catch_rectify_plate_characters(text)
  print(text)
  #print(t)
  return text


print(ocr('./output/temp.jpg'))