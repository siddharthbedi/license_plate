import cv2
import numpy as np
#from PIL import Image

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
    
    cv2.rectangle(img, (int(x),int(y)), (int(x_plus_w),int(y_plus_h)), (255,0,0), 2)

    cv2.putText(img, label, (int(x)-10,int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    




def detect(path, net, classes):
    image = cv2.imread(path)
    if image is None:


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
        cv2.imwrite("./output/cropped/cropped"+str(i)+".jpg", crop_img)

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
    cv2.imwrite('./output/proc.jpg', image)
    img = cv2.imread('./output/proc.jpg')
    

    cv2.waitKey(0)

    return image


if __name__ == '__main__':


  config = './data/vehicle-detector/yolo-voc.cfg'
  weights =  './data/vehicle-detector/yolo-voc.weights'
  classes = './data/vehicle-detector/voc.names'


  with open(classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

  net = cv2.dnn.readNet(weights, config)

  path = './samples/test/*'    #home/siddh/Desktop/PythonData/Numberplate/yolo_liense/alpr-unconstrained

  
  image = detect(path, net, classes)


 