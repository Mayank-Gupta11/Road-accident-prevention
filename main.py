import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer


#haar cascades for face, left eye and right eye
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')


lbl=['Close','Open']

#loading the trained model
model = load_model('models/cnncat2.h5')
path = os.getcwd()

#using opencv to capture a video
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

#initialising the alarm audio file to play it
mixer.init()
sound = mixer.Sound('alarm.wav')

while(True):
	ret, img = cap.read()
	height,width = img.shape[:2]

	#haar cascades require an image in grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
	if len(faces)!=0:#detecting if the driver is present or not
		left_eye = leye.detectMultiScale(gray)
		right_eye =  reye.detectMultiScale(gray)

		cv2.rectangle(img, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

		for (x,y,w,h) in faces:
			#forming a rectangle around the face
			cv2.rectangle(img, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

		for (x,y,w,h) in right_eye:
			r_eye=img[y:y+h,x:x+w]
			count=count+1
			r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
			r_eye = cv2.resize(r_eye,(24,24))
			r_eye= r_eye/255
			r_eye=  r_eye.reshape(24,24,-1)
			r_eye = np.expand_dims(r_eye,axis=0)
			#predicting open or close for the right eye using the model
			rpred = model.predict(r_eye)

			print(rpred[0])
			if(rpred[0][0]>=0.85):
				lbl='Closed'
			if(rpred[0][0]<=0.20):
				lbl='Open'
				sound.stop()
			break

		for (x,y,w,h) in left_eye:
			l_eye=img[y:y+h,x:x+w]
			count=count+1
			l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
			l_eye = cv2.resize(l_eye,(24,24))
			l_eye= l_eye/255
			l_eye=l_eye.reshape(24,24,-1)
			l_eye = np.expand_dims(l_eye,axis=0)
			#predicting open or close for left eye using the model
			lpred = model.predict(l_eye)
			if(lpred[0][0]>=0.85):
				lbl='Closed'
			if(lpred[0][0]<=0.20):
				lbl='Opend'
			break
		#evaluating score based of parameters
		if(rpred[0][0]>=0.80 and lpred[0][0]>=0.80):
			score=score+1
			cv2.putText(img,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
		else:
			score=score-1
			cv2.putText(img,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
			sound.stop()


		if(score<0):
			score=0
		cv2.putText(img,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
		if(score>5):
			#cv2.imwrite(os.path.join(path,'image.jpg'),img)
			try:
				sound.play()
				#print("Wake up")

			except:
				pass
			if(thicc<16):
				thicc= thicc+2
			else:
				thicc=thicc-2
				if(thicc<2):
					thicc=2
			cv2.rectangle(img,(0,0),(width,height),(0,0,255),thicc)

	else:
		sound.stop()
		cv2.putText(img, "NO DRIVER", (int(width/3), int(height/2)),\
		cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
	cv2.imshow('img',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
