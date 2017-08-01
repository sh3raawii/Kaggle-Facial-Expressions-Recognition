# Author: Mostafa Mahmoud Ibrahim Hassan
# Email: mostafa_mahmoud@protonmail.com

import os
import numpy as np
import cv2 as cv
from skvideo.io import FFmpegWriter
from keras.models import load_model
from dlib import get_frontal_face_detector
from utils import square_image

facial_expressions = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")


if __name__ == '__main__':
    # get model path from the user
    model_path = input(">>Please enter the path to the model including the h5 extension "
                       "[for default (./saved_models/my_model.h5) press enter]: ")
    model_path = "./saved_models/my_model.h5" if model_path.isspace() or model_path == "" else model_path
    model_path = os.path.realpath(model_path)
    assert os.path.isfile(model_path)
    # get preferences
    save_webcam_session = input(">>Do you want to save the webcam session ? [Y/N]: ")
    save_webcam_session = True if save_webcam_session.lower() == "y" or save_webcam_session.lower() == "yes" else False
    # load keras model
    fer_model = load_model(model_path)
    # load face detector
    detector = get_frontal_face_detector()
    # set up a video writer
    file_name = "webcam_test.mp4"
    dir_path = os.path.dirname(__file__)
    video_writer = FFmpegWriter(os.path.join(dir_path, file_name))
    # capture webcam feed
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            # get faces in the frame
            detections, confidences, idx = detector.run(frame, 0)
            if len(detections) > 0:
                detection = detections[0]
                left, top, right, bottom = detection.left(), detection.top(), detection.right(), detection.bottom()
                # this is because of dlib bug that gets out of index co-ordinates
                face_crop = frame[max(0, top):min(bottom, frame.shape[0]), max(0, left):min(right, frame.shape[1])]
                # transform to square grey scale image and resize to 48x48
                face_crop, _ = square_image(face_crop)
                face_crop = cv.resize(face_crop, (48, 48))
                face_crop = cv.cvtColor(face_crop, cv.COLOR_BGR2GRAY)
                # reshape, adding dimensions needed
                input_batch = face_crop.reshape([1, 48, 48, 1])
                output_batch = fer_model.predict(input_batch)
                # draw confidence bars on the frame
                idx = np.argmax(output_batch[0])
                for i, (facial_expression_name, confidence) in enumerate(zip(facial_expressions, output_batch[0])):
                    color = (0, 0, 255) if i == idx else (255, 0, 0)
                    cv.putText(frame, facial_expression_name, (10, 20 + i * 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                               color)
                    cv.rectangle(frame, (150, 10 + i * 30), (int(150 + confidence * 200), 10 + i * 30 + 8),
                                 color=color, thickness=-1)
            # display the frame
            cv.imshow("facial expressions", frame)
            if save_webcam_session:
                video_writer.writeFrame(frame[:, :, ::-1])
            if cv.waitKey(1) & 0xFF == ord('q'):
                if save_webcam_session:
                    video_writer.close()
                break
    cap.release()
    cv.destroyAllWindows()
