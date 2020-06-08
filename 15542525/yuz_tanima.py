import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
 

def print_utf8_text(image, xy, text, color):
 
    font =  ImageFont.truetype("arial.ttf", 18)  
    img_pil = Image.fromarray(image)  
    draw = ImageDraw.Draw(img_pil)  
    draw.text((xy[0],xy[1]), text, font=font,
              fill=(color[0], color[1], color[2], 0))  
    image = np.array(img_pil)  
    return image
 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('ogretici/trainer.yml')
cascadePath = "siniflar/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['None', 'Çağdaş','Ahmet','Muhlise','Canay']
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
      
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "bilinmiyor"
            confidence = "  {0}%".format(round(100 - confidence))
 
        color = (255,255,255)
        img=print_utf8_text(img,(x + 5, y - 25),str(id),color)      
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1) 
    cv2.imshow('Yuz Tanima', img)
    k = cv2.waitKey(10) & 0xff  
    if k == 27 or k==ord('q'):
        break
print("\n Programdan çıkılıyor.")
cam.release()
cv2.destroyAllWindows()