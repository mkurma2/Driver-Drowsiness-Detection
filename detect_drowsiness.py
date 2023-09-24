
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav


from scipy.spatial import distance as ssd
from imutils.video import VideoStream as VS
from imutils import face_utils as fu
from threading import Thread as TD
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):

	playsound.playsound(path)

def eye_aspect_ratio(eyes):

	Vertical_eyedistance1 = ssd.euclidean(eyes[1],eyes[5])
	Vertical_eyedistance2 = ssd.euclidean(eyes[2],eyes[4])


	Horizontal_eyedistance2 = ssd.euclidean(eyes[0],eyes[3])


	eye_aspect_ratio_calculated = (Vertical_eyedistance1 + Vertical_eyedistance2) / (2.0 * Horizontal_eyedistance2)


	return eye_aspect_ratio_calculated
 

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())
 

Eye_aspect_ratio_indicator = 0.3
Eye_consecutive_frames = 48


frame_counter = 0
Indicator_of_alarm_is_on = False


print("[1] loading our facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = fu.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = fu.FACIAL_LANDMARKS_IDXS["right_eye"]


print("[2] starting our video stream ....")
vs = VS(src=args["webcam"]).start()
time.sleep(1.0)


while True:

	RNF = vs.read() #readinnextframe
	RNF = imutils.resize(RNF, width=450)
	gray = cv2.cvtColor(RNF, cv2.COLOR_BGR2GRAY)


	face_detects = detector(gray, 0)


	for r in face_detects:

		shape = predictor(gray, r)
		shape = fu.shape_to_np(shape)


		Eye_onleft = shape[lStart:lEnd]
		Eye_onright = shape[rStart:rEnd]
		Ear_onleft = eye_aspect_ratio(Eye_onleft)
		Ear_onright = eye_aspect_ratio(Eye_onright)


		average_eye_aspect_ratio = (Ear_onleft + Ear_onright) / 2.0

		left_eyeconvexhull = cv2.convexHull(Eye_onleft)
		right_eyeconvexhull = cv2.convexHull(Eye_onright)
		cv2.drawContours(RNF, [left_eyeconvexhull], -1, (0, 255, 0), 1)
		cv2.drawContours(RNF, [right_eyeconvexhull], -1, (0, 255, 0), 1)


		if average_eye_aspect_ratio < Eye_aspect_ratio_indicator:
			frame_counter += 1


			if frame_counter >= Eye_consecutive_frames:

				if not Indicator_of_alarm_is_on:
					Indicator_of_alarm_is_on = True


					if args["alarm"] != "":
						t = TD(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()


				cv2.putText(RNF, "DROWSINESS ALERT!", (10, 40),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


		else:
			frame_counter = 0
			Indicator_of_alarm_is_on = False


		cv2.putText(RNF, "Eye aspect ratio: {:.2f}".format(average_eye_aspect_ratio), (260, 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
 

	cv2.imshow("Alert Today , Alive Tomorrow", RNF)
	key = cv2.waitKey(1)
 

	if key == ord("a"):
		break


cv2.destroyAllWindows()
vs.stop()