import os

import cv2 as cv
import face_recognition


def main():
    known_face_encodings = []
    known_face_names = []
    file_dir = os.path.join(os.path.dirname(__file__), 'face_data')
    for file in [i for i in os.listdir(file_dir) if i.endswith(('.jpg', '.png'))]:
        image = face_recognition.load_image_file(os.path.join(file_dir, file))
        face_locations = face_recognition.face_locations(image)
        face_encoding = face_recognition.face_encodings(image, face_locations)

        known_face_encodings.append(face_encoding[0])
        known_face_names.append(file[:file.index('.')])

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        flip = cv.flip(frame, 1)

        face_locations = face_recognition.face_locations(flip)
        if face_locations:
            for face in face_locations:
                start_y, end_x, end_y, start_x = face
                face_encodings = face_recognition.face_encodings(flip, [(start_y, end_x, end_y, start_x)])

                for face_encoding_to_check in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check,
                                                             tolerance=0.55)

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
