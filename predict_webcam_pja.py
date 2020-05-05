from keras.models import load_model
import sys
import cv2
import os
import numpy as np

def set_params():
    params = dict()
    params["num_gpu"] = -1
    params["model_folder"] = "../sitting_posture_project/openpose/models/"
    params["disable_multi_thread"] = True
    params["fps_max"] = -1
    return params
params = set_params()

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../sitting_posture_project/openpose/python/openpose/Release');
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../sitting_posture_project/openpose/x64/Release;' +  dir_path + '/../sitting_posture_project/openpose/bin;'
import pyopenpose as op

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Process and display images
    ret, frame = cap.read()
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    font = cv2.FONT_HERSHEY_SIMPLEX
    if np.size(datum.poseKeypoints) != 75:
        cv2.putText(frame, "show your whole body!!!", (10, 50), font, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        points = datum.poseKeypoints[0, :16, :2]
        points = np.reshape(points, (1, 32))
        if np.count_nonzero(points) != 32:
            cv2.putText(frame, "missing point!!!", (10, 50), font, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            points = points / 2
            model = load_model('sitting_model.h5')
            predictions = model.predict(points)
            if predictions >= 0.3:
                cv2.putText(frame, 'Bad', (10, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Good', (10, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    if cv2.waitKey(1) == ord('q'):
        break
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()

