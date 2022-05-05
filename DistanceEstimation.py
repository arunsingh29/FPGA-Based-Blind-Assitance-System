import cv2 as cv 
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 45 #INCHES
PERSON_WIDTH = 16 #INCHES
MOBILE_WIDTH = 3.0 #INCHES

# Object detector constant 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)
# defining fonts 
FONTS = cv.FONT_HERSHEY_SIMPLEX

# getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id 
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid[0]], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
        #print('BOX:',box)
    
        # getting the data 
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2),box[3]])
        elif classid ==67: # cell phone
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1]-2),box[3]])
        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data. 
    return data_list

def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

# distance finder function 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance

# reading the reference image from dir 
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

#print('Joshy',person_data)
#print('width',person_data[0][1])
print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

# finding focal length 
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    for d in data:
        #print(d)
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='cell phone':
            distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        
        cv.rectangle(frame, (x, y-3), (x+150, y+83),BLACK,-1 )
        cv.putText(frame, f'Dis: {round(distance,2)} inch', (x+5,y+13), FONTS, 0.48, GREEN, 2)
        cv.putText(frame, f'Dis: {round(distance*2.54,2)} cm', (x+5,y+30), FONTS, 0.48, GREEN, 2)
        cv.putText(frame, f'Xc: {round((x+d[1])/2,2)}', (x+5,y+50), FONTS, 0.48, GREEN, 2) #x=x1 y=y1 d[1]=x2 d[3]=y2
        cv.putText(frame, f'Yc: {round((y+2+d[3])/2,2)}', (x+5,y+70), FONTS, 0.48, GREEN, 2)
        print(d[0])
        print(f'Xc: {round((x+d[1])/2,2)}')
        print(f'Yc: {round((y+2+d[3])/2,2)}')
        print(f'Dis: {round(distance*2.54,2)} cm')
        #print(frame.shape)
        if(round((x+d[1])/2,2)<= 170):
            print('left')
        elif(round((x+d[1])/2,2)<= 312):
             print('centre')
        else:
            print("Right")
    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

