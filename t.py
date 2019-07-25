import cv2
import numpy as np
#from PIL import Image
import os, sys
from src.keras_utils import load_model
from src.keras_utils import load_model, detect_lp
import traceback

from src.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes

from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import os, sys
import tesserocr


config = './data/vehicle-detector/yolo-voc.cfg'
weights =  './data/vehicle-detector/yolo-voc.weights'
classes = './data/vehicle-detector/voc.names'
wpod_net_path = './data/lp-detector/wpod-net_update1.h5'
lp_threshold = .5

def adjust_pts(pts,lroi):
  return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))




def getOutputsNames(net):
    layersNames = net.getLayerNames()
    layer = []
    for i in net.getUnconnectedOutLayers():
        layer.append(layersNames[i[0] - 1])

       # print(i[0]-1)
    #return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return layer

# Darw a rectangle surrounding the object and its class name 
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    #olor = COLORS[class_id]
    
    cv2.rectangle(img, (int(x),int(y)), (int(x_plus_w),int(y_plus_h)), (255,0,0), 1)

    cv2.putText(img, label, (int(x)-10,int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    

def detect_license_plate(Ivehicle, lp_threshold, wpod_net):

  
  try:

    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
    side  = int(ratio*288.)
    bound_dim = min(side + (side%(2**4)),608)
    #print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)

    Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

    if len(LlpImgs):
      Ilp = LlpImgs[0]
      Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
      Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
      return Ilp
  

  except:
    pass
      

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
  text = pytesseract.image_to_string(Image.open('./output/temp.jpg'), config="-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz -psm 6")
  t = tesserocr.image_to_text(Image.open('./output/temp.jpg'))
  #os.remove('./output/temp.jpg')
  print(text)
  print(t)
  return text


def detect(image, net, classes):
    #image = cv2.imread(path)
    if image is None:

      print('blah')
      return "Image not found"

    else:
      blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
      Width = image.shape[1]
      Height = image.shape[0]
      net.setInput(blob)
      outs = net.forward(getOutputsNames(net))

      class_ids = []
      confidences = []
      boxes = []
      conf_threshold = 0.5              #0.5
      nms_threshold = 0.4                 #0.4


   #print(len(outs))

   # In case of tiny YOLOv3 we have 2 output(outs) from 2 different scales [3 bounding box per each scale]
   # For normal normal YOLOv3 we have 3 output(outs) from 3 different scales [3 bounding box per each scale]

   # For tiny YOLOv3, the first output will be 507x6 = 13x13x18
   # 18=3*(4+1+1) 4 boundingbox offsets, 1 objectness prediction, and 1 class score.
   # and the second output will be = 2028x6=26x26x18 (18=3*6) 

    for out in outs: 
       #print(out.shape)
        for detection in out:
           
       #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
           scores = detection[5:]#classes scores starts from index 5
           class_id = np.argmax(scores)
           confidence = scores[class_id]
           #print(confidence)
           if confidence > 0.5: 
              
              #0.5
              center_x = int(detection[0] * Width)
              center_y = int(detection[1] * Height)
              w = int(detection[2] * Width)
              h = int(detection[3] * Height)
              x = center_x - w / 2
              y = center_y - h / 2
              class_ids.append(class_id)
              confidences.append(float(confidence))
              boxes.append([x, y, w, h])
               #print(boxes)

   # apply  non-maximum suppression algorithm on the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

   #print(boxes)
    if len(boxes) == 0:

      flag = False

    if len(boxes) != 0:
      flag = True


   
    crop_imgs = {}
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        #print(i)
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        crop_img = image[round(y):round(y+h), round(x):round(x+w)]

        license_plate = detect_license_plate(crop_img, lp_threshold, wpod_net)
        #cv2.imshow('ll', license_plate)
        #cv2.waitKey(0)

        try:
          path = "./output/license/"+str(i)+".png"
          cv2.imwrite(path, license_plate*255.)

        #crop_imgs.append(crop_img)
        #cv2.imwrite("./output/cropped/cropped"+str(i)+".jpg", crop_img)
          number = ocr(path)

          if number is not None:

            print(number)
            cv2.putText(image, number, (round(x+h/2),round(y+w/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        except:
          pass


    # Put efficiency information.
    #t, _ = net.getPerfProfile()
    #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    #cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    r_img = cv2.resize(img, (640, 480))
    cv2.imshow('./output/proc.jpg', r_img)
    #img = cv2.imread('./output/proc.jpg')
    

    #cv2.waitKey(0)

    return image

if __name__ == '__main__':
  wpod_net = load_model(wpod_net_path)
  with open(classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

  net = cv2.dnn.readNet(weights, config)

  path = 'test_cut.mp4' 
  video_file = 'video.mp4'

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('processed.mp4', fourcc, 20.0, (640, 480))

  cap = cv2.VideoCapture(path)


  while True:
    ret, img = cap.read()

    if ret == True:
      rows = img.shape[0]
      cols = img.shape[1]

      M = cv2.getRotationMatrix2D((cols/2 , rows/2), -90 ,1)
      #img = cv2.warpAffine(img,M,(cols , rows)) 
      image = detect(img, net, classes)
      out.write(image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      cap.release()
      cv2.destroyAllWindows()
      out.release()
      break

     #home/siddh/Desktop/PythonData/Numberplate/yolo_liense/alpr-unconstrained
'''  path = "./samples/test"
  dirs = os.listdir( path )
  for item in dirs:
    print(item)


    im = path+'/'+item
    print(im)
    crop_imgs = detect(im, net, classes)'''




 
