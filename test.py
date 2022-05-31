import cv2
import PoseModule as pm

img = cv2.imread("Pose/2.jpg")
img = cv2.resize(img, (500, 500))

# Create athlete
athlete = pm.poseDetector()

# Detect athlete
img = athlete.findPose(img)
# Get landmarks
lmList = athlete.findPosition(img)
# Position athlete in the center of the image
cv2.putText(img, athlete.inPicture(img), (10, 900), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

# Find Angles
if len(lmList) != 0:
    print(athlete.findAngle(img, 11, 13, 15))

cv2.imshow('ATHLETTO', img)
cv2.waitKey(0)
