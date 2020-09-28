import cv2 
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x + h]
        roi_color = frame[y:y + h, x:x + h]
        id_, conf = recognizer.predict(roi_gray)
        if conf <= 40:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            storke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, storke, cv2.LINE_AA)
        #img_item = "7.png"
        #cv2.imwrite(img_item, roi_color)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

