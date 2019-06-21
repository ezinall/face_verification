import os

import cv2 as cv
import dlib
import numpy as np
import face_recognition
from scipy.spatial import distance


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def main():
    cap = cv.VideoCapture(0)

    # net = cv.dnn.readNetFromTensorflow('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
    # net = cv.dnn.readNetFromCaffe('daploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    known_face_encodings = []
    known_face_names = []
    file_dir = os.path.join(os.path.dirname(__file__), 'face_data')
    for file in [i for i in os.listdir(file_dir) if i.endswith(('.jpg', '.png'))]:
        image = face_recognition.load_image_file(os.path.join(file_dir, file))
        face_locations = face_recognition.face_locations(image)
        face_encoding = face_recognition.face_encodings(image, face_locations)

        known_face_encodings.append(face_encoding[0])
        known_face_names.append(file[:file.index('.')])

    while True:
        ret, frame = cap.read()

        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flip = cv.flip(frame, 1)

        # (h, w) = flip.shape[:2]
        # blob = cv.dnn.blobFromImage(cv.resize(flip, (300, 300)), 1.0,
        #                             (300, 300), (103.93, 116.77, 123.68))
        # net.setInput(blob)
        # detections = net.forward()
        #
        # for i in range(detections.shape[2]):
        #     confidence = detections[0, 0, i, 2]
        #
        #     if confidence > 0.95:
        #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        #         (startX, startY, endX, endY) = box.astype("int")
        #
        #         cv.rectangle(flip, (startX, startY), (endX, endY),
        #                      (0, 0, 255), 2)
        #
        #         # text = "{:.2f}%".format(confidence * 100)
        #         # y = startY - 10 if startY - 10 > 10 else startY + 10
        #         # cv.putText(flip, text, (startX, y),
        #         #            cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #
        #         # shape = shape_to_np(shape)
        #         # for (x, y) in shape:
        #         #     cv.circle(flip, (x, y), 1, (0, 255, 0), -1)
        #
        #         # rect_d = dlib.rectangle(startX, startY, endX, endY)
        #         # shape = predictor(flip, rect_d)
        #         # face_descriptor = facerec.compute_face_descriptor(flip, shape)
        #
        #         # small_frame = cv.resize(flip, (0, 0), fx=0.25, fy=0.25)
        #         face_locations = face_recognition.face_locations(flip)
        #         if face_locations:
        #             startY, endX, endY, startX = face_locations[0]
        #             cv.rectangle(flip, (startX, startY), (endX, endY), (0, 255, 0), 2)
        #         face_encodings = face_recognition.face_encodings(flip, [(startY, endX, endY, startX)])
        #
        #         for face_encoding_to_check in face_encodings:
        #             matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check,
        #                                                      tolerance=0.6)
        #
        #             if True in matches:
        #                 print(known_face_names[matches.index(True)])
        #
        #                 text = "{}".format(known_face_names[matches.index(True)])
        #                 y = startY - 10 if startY - 10 > 10 else startY + 10
        #                 cv.putText(flip, text, (startX, y),
        #                            cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        #             else:
        #                 known_face_encodings.append(face_encoding_to_check)
        #                 name = 'Person {}'.format(len(known_face_encodings))
        #                 known_face_names.append(name)
        #                 cv.imwrite(os.path.join(file_dir, f'{name}.png'), flip[startY:endY, startX:endX])

        face_locations = face_recognition.face_locations(flip)
        if face_locations:
            for face in face_locations:
                start_y, end_x, end_y, start_x = face
                face_encodings = face_recognition.face_encodings(flip, [(start_y, end_x, end_y, start_x)])

                for face_encoding_to_check in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check,
                                                             tolerance=0.6)

                    if True in matches:
                        print(known_face_names[matches.index(True)])

                        text = "{}".format(known_face_names[matches.index(True)])
                        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
                        cv.putText(flip, text, (start_x, y),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    else:
                        known_face_encodings.append(face_encoding_to_check)
                        name = 'Person {}'.format(len(known_face_encodings))
                        known_face_names.append(name)
                        cv.imwrite(os.path.join(file_dir, f'{name}.png'), flip[start_y:end_y, start_x:end_x])

                cv.rectangle(flip, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        cv.imshow('frame', flip)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
