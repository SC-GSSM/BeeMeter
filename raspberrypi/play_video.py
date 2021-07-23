import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

video = cv2.VideoCapture(args.filename)

while True:
    success, frame = video.read()
    if success:
        cv2.imshow("Video", frame)
        cv2.waitKey(30)
    else:
        break
