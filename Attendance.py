import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import  datetime
print("WELCOME! This is a Face detection test.Please stay still in front of camera.Otherwise go to hell :)")
print("Scanning face.")
print("Working on Detection...")

path = "C:\\Users\\Shreyas\\PYTHON PROGRAMMING YOUTUBE 12HRS COURSE\\images"
images = []
className = []
myList = os.listdir(path)
# print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    className.append(os.path.splitext(cls)[0])
# print(className)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded)
    return encodeList

def AttenList(name):
    with open("Attendance.csv","r+") as f:
        mydatalist=f.readlines()
        nameListAtten = []
        for line in mydatalist:
            entry=line.split(",")
            nameListAtten.append(entry[0])
        if name not in nameListAtten:
            now=datetime.now()
            dstring=now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dstring}')




encodelistKnown = findEncodings(images)
print("Encoding done")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodingCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encFace, FaceLoc in zip(encodingCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodelistKnown, encFace)
        faceDis = face_recognition.face_distance(encodelistKnown, encFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            # print(name)
            time.sleep(1.5)
            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255),
                        2)  # 1 is font thickness
            AttenList(name)
        else:
            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 255, 0), cv2.FILLED)
            cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
            # print("NO MATCH!!!")
            time.sleep(1.5)
    cv2.imshow("Camera", img)
    cv2.waitKey(1)
