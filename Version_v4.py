from flask import Flask, render_template,Response
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import dlib
from scipy.spatial import distance as dist

global camera
app = Flask(__name__)
class AdventureDone(Exception): pass

# known_face_encodings = np.load('encodeListKnown_test.npy', allow_pickle=True)
# known_face_encodings = np.load('D:/Projects/Internal Projects/Attendance System/Version_1/encodeListKnown_test_v5.npy', allow_pickle=True)
known_face_encodings = np.load('encodeListKnown_test_v5.npy', allow_pickle=True)
#known_face_names = ['Abhishek J', 'Ansh M', 'Aashi G ','Aashi G', 'Aashi G', 'Aashi G', 'Barkha S', 'Madan A', 'Madan A', 'Manish P', 'Manish P', 'Manish P', 'Manish P', 'Manish P', 'Manvi', 'Manvi', 'Navdeep', 'Navdeep', 'Navdeep', 'Pankaj Dwivedi', 'Parv Yadav', 'Parv Yadav', 'Rohit Bamoriya', 'Sanjay Gurjar', 'Shreya Goyal', 'Shreya Goyal', 'Tanvi Bhave', 'Tarun Sinhal', 'Tarun Sinhal', 'Tilottama Sharma', 'Vandana Chouhan', 'Vijay Patidar', 'Vijeet Agrawal']
#known_face_names = ['Abhishek J', 'Ansh M', 'Ashi G', 'Ashi G', 'Ashi G', 'Ashi G', 'Barkha S', 'Bhaneshwari', 'Bhaneshwari', 'Bhaneshwari', 'Deepak', 'Deepak', 'Deepak', 'Madan A', 'Madan A', 'Manish P', 'Manish P', 'Manish P', 'Manish P', 'Manish P', 'Manvi', 'Manvi', 'Navdeep G', 'Navdeep G', 'Navdeep G', 'Pankaj D', 'Parv Y', 'Parv Y', 'Priyank P', 'Priyank P', 'Priyank P', 'Rohit B', 'Sanjay G', 'Shreya G', 'Shreya G', 'Tanvi B', 'Tarun S', 'Tarun S', 'Tilottama S', 'Vandana C', 'Vijay P', 'Vijeet A']
#known_face_names = ['Abhishek', 'Ansh', 'Ashi', 'Ashi', 'Ashi', 'Ashi','Ashish','Ashish','Ashish','Ashish','Ashish', 'Barkha', 'Barkha', 'Barkha', 'Barkha', 'Barkha', 'Bhaneshwari', 'Bhaneshwari', 'Bhaneshwari', 'Deepak', 'Deepak', 'Deepak', 'Madan', 'Madan', 'Manish', 'Manish', 'Manish', 'Manish', 'Manish', 'Manvi', 'Manvi', 'Navdeep', 'Navdeep', 'Navdeep', 'Pankaj', 'Parv', 'Parv', 'Payal', 'Payal', 'Payal', 'Pooja', 'Pooja', 'Pooja', 'Priyank', 'Priyank', 'Priyank', 'Rohit', 'Sanjay', 'Shivangi', 'Shivangi', 'Shivangi', 'Shivangi', 'Shivangi','Shraddha', 'Shraddha', 'Shraddha', 'Shraddha', 'Shraddha', 'Shreya', 'Shreya','Tanishka','Tanishka','Tanishka','Tanishka', 'Tanvi', 'Tarun', 'Tarun', 'Tilottama', 'Vandana', 'Vijay', 'Vijeet']
known_face_names = ['Aman', 'Aman', 'Aman', 'Aman', 'Aman', 'Aman', 'Aman', 'Ashi', 'Ashi', 'Ashi', 'Ashi', 'Ashish', 'Ashish', 'Ashish', 'Ashish', 'Ashish', 'Barkha Sharma', 'Barkha Sharma', 'Barkha Sharma', 'Barkha Sharma', 'Barkha Sharma', 'Bhaneshwari', 'Bhaneshwari', 'Bhaneshwari', 'Chhaya', 'Chhaya', 'Chhaya', 'Chhaya', 'Chhaya', 'Deepak', 'Deepak', 'Deepak', 'Madan Agrawal', 'Madan', 'Madhav', 'Madhav', 'Madhav', 'Madhav', 'Madhav', 'Manvi', 'Manvi', 'Navdeep', 'Navdeep', 'Navdeep', 'Nikita ', 'Nikita', 'Nikita', 'Nikita', 'Nikita', 'Nikita', 'Nikita', 'Pankaj Dwivedi', 'Parv Yadav', 'Parv', 'Payal', 'Payal', 'Payal', 'Pooja', 'Pooja', 'Pooja', 'Prashasht', 'Prashasht', 'Prashasht', 'Prashasht', 'Prashasht', 'Prashasht', 'Prashasht',
 'Prashasht', 'Priyank', 'Priyank', 'Priyank', 'Shivangi', 'Shivangi', 'Shivangi', 'Shivangi', 'Shivangi', 'Shraddha', 'Shraddha', 'Shraddha', 'Shraddha', 'Shraddha', 'Shreya', 'Shreya Goyal', 'Tanishka', 'Tanishka', 'Tanishka', 'Tanishka', 'Tanvi Bhave', 'Tarun', 'Tarun Sinhal', 'Tilottama Sharma', 'Vandana Chouhan', 'Vijay Patidar', 'Vijeet Agrawal']


def eye_aspect_ratio(eye):
    # compute the euclidean distance between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # compute the EAR
    ear = (A + B) / (2 * C)
    return ear

def markAttendance(name,camera,app,loggedin_loggedout):
    print('Inside Mark Attendance')
    with app.app_context(), app.test_request_context():
        # with open('D:/Projects/Internal Projects/Attendance System/Version_1/Attendance.csv', 'r+') as f:
        with open('Attendance.csv', 'r+') as f:
            print('Inside Attendance csv')
            now = datetime.now()
            global dtString
            dtString = now.strftime('%H:%M:%S')
            dString = now.strftime('%d:%b:%Y')
            myDataList = f.readlines()
            nameList = []
            dtStringList = []
            nameListCurrent = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
                dtStringList.append(entry[1])
            if name not in nameList:
                print('Inside adding Attendance csv')
                f.writelines(f'\n{name},{dtString},{dString},{loggedin_loggedout}')
                print(1)
                camera.release()
                return render_template('cam.html',prediction_text="{}".format(name))
            elif dtStringList[nameList.index(name)] != dtString:
                if name not in nameListCurrent:
                    print('Inside adding 2 Attendance csv')
                    f.writelines(f'\n{name},{dtString},{dString},{loggedin_loggedout}')
                    nameListCurrent.append(name)
                    camera.release()
                    cv2.destroyAllWindows()
                    print(2)
                    return render_template('cam.html',prediction_text="{}".format(name))
            elif name in nameList:
                raise AdventureDone
def gen_framesout(app):
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    JAWLINE_POINTS = list(range(0, 17))
    RIGHT_EYEBROW_POINTS = list(range(17, 22))
    LEFT_EYEBROW_POINTS = list(range(22, 27))
    NOSE_POINTS = list(range(27, 36))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_OUTLINE_POINTS = list(range(48, 61))
    MOUTH_INNER_POINTS = list(range(61, 68))

    EYE_AR_THRESH = 0.33
    EYE_AR_CONSEC_FRAMES = 2
    EAR_AVG = 0

    COUNTER = 0
    TOTAL = 0

    # to detect the facial region
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks (2).dat')

    camera = cv2.VideoCapture(0)
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))
    size = (frame_width, frame_height)
    now = datetime.now()
    now = now.strftime('%H_%M_%S')
    print(type(now))
    # result = cv2.VideoWriter('D:/Python Face Recognition/Web App v3/VideoDataset/{}.avi'.format(now),
    #                          cv2.VideoWriter_fourcc(*'MJPG'),
    #                          240, size)
    sequence_count_frame = 0
    try:
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                # result.write(frame)
                sequence_count_frame += 1
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                #########################
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)

                #########################
                for rect in rects:
                    x = rect.left()
                    y = rect.top()
                    x1 = rect.right()
                    y1 = rect.bottom()
                    # get the facial landmarks
                    landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
                    # get the left eye landmarks
                    left_eye = landmarks[LEFT_EYE_POINTS]
                    # get the right eye landmarks
                    right_eye = landmarks[RIGHT_EYE_POINTS]
                    # draw contours on the eyes
                    left_eye_hull = cv2.convexHull(left_eye)
                    right_eye_hull = cv2.convexHull(right_eye)
                    cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0),
                                     1)  # (image, [contour], all_contours, color, thickness)
                    cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                    # compute the EAR for the left eye
                    ear_left = eye_aspect_ratio(left_eye)
                    # compute the EAR for the right eye
                    ear_right = eye_aspect_ratio(right_eye)
                    # compute the average EAR
                    # ear_avg = ear_left
                    ear_avg = (ear_left + ear_right) / 2.0
                    # detect the eye blink
                    # if ear_avg < EYE_AR_THRESH:
                    if ear_left < EYE_AR_THRESH or ear_right < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                            print("Eye blinked")
                        COUNTER = 0

                    cv2.putText(frame, "Winked {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0),
                                1)
                    #cv2.putText(frame, "EAR {}".format(ear_avg), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255),1)
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                        name = "Unknown"
                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            face_names.append(name)
                        # Display the results
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            # Draw a box around the face
                            #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            now = datetime.now()
                            now = now.strftime('%H:%M')
                            putText = 'Time ' + now

                            # Draw a label with a name below the face
                            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX

                            #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                            #cv2.putText(frame, putText, (450, 468), font, 1.0, (0, 0, 0), 2)
                            if sequence_count_frame >= 15 and TOTAL >= 1:
                                cv2.putText(frame, name + ' Logged Out ', (200,268), font, 1.0, (0, 0, 0), 2)
                                cv2.putText(frame, 'at ' + putText, (200, 300), font, 1.0, (0, 0, 0), 2)
                                loggedin_loggedout = 'Logged Out'
                                markAttendance(name,camera,app,loggedin_loggedout)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except AdventureDone:
        pass
def gen_frames(app):
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    JAWLINE_POINTS = list(range(0, 17))
    RIGHT_EYEBROW_POINTS = list(range(17, 22))
    LEFT_EYEBROW_POINTS = list(range(22, 27))
    NOSE_POINTS = list(range(27, 36))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    MOUTH_OUTLINE_POINTS = list(range(48, 61))
    MOUTH_INNER_POINTS = list(range(61, 68))

    EYE_AR_THRESH = 0.33
    EYE_AR_CONSEC_FRAMES = 2
    EAR_AVG = 0

    COUNTER = 0
    TOTAL = 0

    # to detect the facial region
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks (2).dat')

    camera = cv2.VideoCapture(0)
    frame_width = int(camera.get(3))
    frame_height = int(camera.get(4))
    size = (frame_width, frame_height)
    now = datetime.now()
    now = now.strftime('%H_%M_%S')
    print(type(now))
    # result = cv2.VideoWriter('D:/Face_Distortion/live_eye_faceRecognition/Live_face_v6/VideoDataset/{}.avi'.format(now),
    #                          cv2.VideoWriter_fourcc(*'MJPG'),
    #                          result                      240, size)
    sequence_count_frame = 0
    try:
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                # result.write(frame)
                sequence_count_frame += 1
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                #########################
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)
                for rect in rects:
                    x = rect.left()
                    y = rect.top()
                    x1 = rect.right()
                    y1 = rect.bottom()
                    # get the facial landmarks
                    landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
                    # get the left eye landmarks
                    left_eye = landmarks[LEFT_EYE_POINTS]
                    # get the right eye landmarks
                    right_eye = landmarks[RIGHT_EYE_POINTS]
                    # draw contours on the eyes
                    left_eye_hull = cv2.convexHull(left_eye)
                    right_eye_hull = cv2.convexHull(right_eye)
                    cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0),
                                     1)  # (image, [contour], all_contours, color, thickness)
                    cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                    # compute the EAR for the left eye
                    ear_left = eye_aspect_ratio(left_eye)
                    # compute the EAR for the right eye
                    ear_right = eye_aspect_ratio(right_eye)
                    # compute the average EAR
                    # ear_avg = ear_left
                    ear_avg = (ear_left + ear_right) / 2.0
                    # detect the eye blink
                    # if ear_avg < EYE_AR_THRESH:
                    if ear_left < EYE_AR_THRESH or ear_right < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                            print("Eye blinked")
                        COUNTER = 0

                    cv2.putText(frame, "Winked {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0),
                                1)
                    #cv2.putText(frame, "EAR {}".format(ear_avg), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255),1)
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                        name = "Unknown"
                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            face_names.append(name)
                        # Display the results
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            # Draw a box around the face
                            #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            now = datetime.now()
                            now = now.strftime('%H:%M')
                            putText = 'Time ' + now

                            # Draw a label with a name below the face
                            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX

                            #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                            #cv2.putText(frame, putText, (450, 468), font, 1.0, (0, 0, 0), 2)
                            if sequence_count_frame >= 15 and TOTAL >= 1:
                                cv2.putText(frame, name + ' Logged IN', (200,268), font, 1.0, (0, 0, 0), 2)
                                cv2.putText(frame, 'at ' + putText, (200,300), font, 1.0, (0, 0, 0), 2)
                                loggedin_loggedout = 'Logged In'
                                print(name)
                                markAttendance(name,camera,app,loggedin_loggedout)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except AdventureDone:
        pass
#python Version_v2.py
#Bhaneshwari
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cam')

def cam():
    return render_template('cam.html')


@app.route('/video_feed')
def video_feed():

    return Response(gen_frames(app), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return 'Hello'

@app.route('/camout')

def camout():
    return render_template('camout.html')
@app.route('/video_feedout')
def video_feedout():

    return Response(gen_framesout(app), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return 'Hello'

if __name__ == '__main__':
    app.run(debug=True)

#python Version_v4.py