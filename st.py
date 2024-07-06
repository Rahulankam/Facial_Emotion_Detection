import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2


facec = cv2.CascadeClassifier('Emotion_detection-main/haarcascade_frontalface_default.xml')

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
        print("Predicted Emotion is" + pred)
        st.write(pred)
    except:
        st.write('unable to detect')

selectbox = st.sidebar.selectbox(
    "Select one of the model",
    ("ferModel", "FacModel")
)


if selectbox == "ferModel":
    model = FacialExpressionModel('Emotion_detection-main/model_a1.json','Emotion_detection-main/model_weights1.h5')
    img = st.file_uploader('Upload image',type=['JPEG','PNG','JPG'])
    st.image(img)
    if img is not None:
        file_img = {"FileName":img.name,"Filetype":img.type}
    
    with open('uploaded_img'+'/'+img.name,'wb') as f:
        f.write(img.getbuffer())
    Detect(img.name)
elif selectbox == 'FacModel':

    imge = st.file_uploader('Upload image',type=['JPEG','PNG','JPG'])
    st.image(imge)
    if imge is not None:
        file_img = {"FileName":imge.name,"Filetype":imge.type}
    
    with open(imge.name,'wb') as f:
        f.write(imge.getbuffer())
    
    def Detect(file_path):
        labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image,1.3,5)
        try:
            for (x,y,w,h) in faces:
                fc = gray_image[y:y+h,x:x+w]
                roi = cv2.resize(fc,(48,48))
                pred = labels[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
            print("Predicted Emotion is" + pred)
            st.write(pred)
        except:
            st.write('unable to detect')

    model = FacialExpressionModel('images/emotiondetector.json','images/emotiondetector.h5')
    
    Detect(imge.name)
else:
    st.write("Select on them")