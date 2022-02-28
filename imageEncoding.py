import cv2
import numpy as np
import face_recognition
import os

path = 'ImagesTraining/idCard'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

#p = ['Abhishek', 'Abhishek', 'Abhishek', 'Abhishek', 'Abhishek', 'Ansh', 'Ansh', 'Barkha', 'Madan', 'Madan', 'Manish', 'Manish', 'Manish', 'Manish', 'Manish', 'Manish', 'Manish', 'Pankaj', 'Pankaj', 'Pankaj', 'Pankaj', 'Pankaj', 'Pankaj', 'Pankaj', 'Pankaj', 'Parv', 'Parv', 'Parv', 'Rohit', 'Sanjay', 'Shreya', 'Shreya', 'Shreya', 'Shreya', 'Shreya', 'Shreya', 'Shreya', 'Shreya', 'Shreya', 'Tanvi', 'Tanvi', 'Tanvi', 'Tanvi', 'Tanvi', 'Tanvi', 'Tanvi', 'Tanvi', 'Tanvi', 'Tarun', 'Tarun', 'Tarun', 'Tilottama', 'Vandana', 'Vandana', 'Vandana', 'Vandana', 'Vandana', 'Vandana', 'Vandana', 'Vijay', 'Vijay', 'Vijay', 'Vijay', 'Vijeet', 'Vijeet', 'Vijeet']
#p = ['Abhishek Joshi', 'Ansh Modi', 'Ashi ', 'Ashi', 'Ashi', 'Barkha Sharma', 'Madan Agrawal', 'Madan', 'Manish Patidar', 'Manish Patidar', 'Manish Patidar', 'Manish Patidar', 'Manish Patidar', 'Manvi', 'Manvi', 'Navdeep', 'Navdeep', 'Navdeep', 'Pankaj Dwivedi', 'Parv Yadav', 'Parv', 'Rohit Bamoriya', 'Sanjay Gurjar','Shreya Goyal', 'Shreya Goyal', 'Tanvi Bhave', 'Tarun Sinhal', 'Tilottama Sharma', 'Vandana Chouhan', 'Vijay Patidar', 'Vijeet Agrawal']
#p = ['Abhishek J', 'Ansh M', 'Ashi G', 'Ashi G', 'Ashi G', 'Ashi G', 'Barkha S', 'Bhaneshwari', 'Bhaneshwari', 'Bhaneshwari', 'Deepak', 'Deepak', 'Deepak', 'Madan P', 'Madan A', 'Manish P', 'Manish P', 'Manish P', 'Manish P', 'Manish P', 'Manvi', 'Manvi', 'Navdeep G', 'Navdeep G', 'Navdeep G', 'Pankaj D', 'Parv Y', 'Parv Y', 'Priyank P', 'Priyank P', 'Priyank P', 'Rohit B', 'Sanjay G', 'Shreya G', 'Shreya G', 'Tanvi B', 'Tarun S', 'Tarun S', 'Tilottama S', 'Vandana C', 'Vijay P', 'Vijeet A']
p = ['Abhishek J', 'Ansh M', 'Ashi', 'Ashi', 'Ashi', 'Ashi', 'Barkha S', 'Bhaneshwari', 'Bhaneshwari', 'Bhaneshwari', 'Deepak', 'Deepak', 'Deepak', 'Madan A', 'Madan A', 'Manish P', 'Manish P', 'Manish P', 'Manish P', 'Manish P', 'Manvi', 'Manvi', 'Navdeep', 'Navdeep', 'Navdeep', 'Pankaj D', 'Parv', 'Parv', 'Pooja', 'Pooja', 'Pooja', 'Priyank P', 'Priyank P', 'Priyank P', 'Rohit B', 'Sanjay G', 'Shreya G', 'Shreya G', 'Tanvi B', 'Tarun S', 'Tarun S', 'Tilottama S', 'Vandana C', 'Vijay P', 'Vijeet A']
def findEncodings(images):
    encodeList = []
    for img in images:
        print(len(encodeList))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
#python imageEncoding.py
encodeListKnown = findEncodings(images)
print('Encoding Complete')

print(encodeListKnown)

np.save('encodeListKnown_test_v5.npy', encodeListKnown, allow_pickle=True)

#python imageEncoding.py

