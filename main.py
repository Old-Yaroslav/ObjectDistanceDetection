import cv2
import cvzone
import torch
import numpy as np
import logging

from time import time
from ultralytics import YOLO
from cvzone.FaceMeshModule import FaceMeshDetector


class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

    def load_model(self):
        model = YOLO("yolov8n.pt")
        model.fuse()

        return model

    def predict(self, frame):
        results = self.model(frame)

        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.cpu().numpy()

            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        frame = results[0].plot()
        return frame, xyxys, confidences, class_ids

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        face_detector = FaceMeshDetector(maxFaces=1)

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret

            operatorMenu = np.zeros_like(frame)

            frame, faces = face_detector.findFaceMesh(frame, draw=False)

            if faces:
                face = faces[0]
                pointLeft = face[145]
                pointRight = face[374]

                # Draw Eyes
                # cv2.line(frame, pointLeft, pointRight, (0, 200, 0), 3)
                # cv2.circle(frame, pointLeft, 5, (255, 0, 255), cv2.FILLED)
                # cv2.circle(frame, pointRight, 5, (255, 0, 255), cv2.FILLED)

                width_in_pixels, _ = face_detector.findDistance(pointLeft, pointRight)
                head_with = 6.3

                # Distance Calculate Here
                # distance_to_object = 30 CHANGEABLE
                # focal_length = (width_in_pixels*distance_to_object)/head_with
                # print(focal_length)  # 867

                focal_length = 867
                distance_to_object = (head_with*focal_length)/width_in_pixels
                print(f'The distance is {int(distance_to_object)}cm')

                # cvzone.putTextRect(frame,
                #                    f'Depth: {int(distance_to_object)}cm',
                #                    (face[10][0]-100, face[10][1]-50),
                #                    scale=2)

                if distance_to_object < 60:
                    cv2.putText(operatorMenu, f'The person is in {int(distance_to_object)}cm', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    pass

            else:
                cv2.putText(operatorMenu, 'No persons', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            results = self.predict(frame)
            frame, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            imgStacked = cvzone.stackImages([frame, operatorMenu], 2, 1)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0))
            cv2.imshow('operator', imgStacked)

            # Press Esc to stop
            if cv2.waitKey(5) & 0xFF == 27:
                print("The program has stopped")
                break

        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index=0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        detector()
    except KeyboardInterrupt:
        print("Forced stop")
        pass
