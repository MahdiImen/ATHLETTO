import cv2
import time
import imutils
import PoseModule as pm


# Define the YOGA poses
# Corrector of pose num 1
def planche(athlete, img):
    if abs(int(athlete.findAngle(img, 25, 23, 26))) not in range(82, 98):
        cv2.putText(img, "Lift your right leg", (50, 800), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif abs(int(athlete.findAngle(img, 23, 25, 27))) not in range(178, 188):
        cv2.putText(img, "Straighten your right leg", (50, 800), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Hold the pose for ten seconds", (50, 800), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return img


# Corrector of pose num 2

def warrior(athlete, img):
    if int(athlete.findAngle(img, 23, 25, 27)) not in range(215, 235):
        cv2.putText(img, "Bend your right foot to reach 230degree", (5, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, "and slide with your left leg ", (5, 780), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif int(athlete.findAngle(img, 31, 27, 29)) not in range(88, 95):
        cv2.putText(img, "Your right foot should face the right", (5, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif int(athlete.findAngle(img, 30, 28, 32)) > 30:
        cv2.putText(img, "Your left foot should face forward", (5, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif int(athlete.findAngle(img, 25, 23, 26)) not in range(110, 120):
        cv2.putText(img, "Slide to keep your right thigh", (5, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, "parallel to the floor", (5, 780), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif int(athlete.findAngle(img, 24, 26, 28)) not in range(175, 190):
        cv2.putText(img, "Keep your left leg straight", (5, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    elif (abs(int(athlete.findAngle(img, 11, 13, 15))) not in range(165, 190)) or (
            abs(int(athlete.findAngle(img, 12, 11, 13))) not in range(165, 190)) or (
            abs(int(athlete.findAngle(img, 12, 14, 16))) not in range(165, 190)) or (
            abs(int(athlete.findAngle(img, 11, 12, 14))) not in range(165, 190)):
        cv2.putText(img, "Hold up your arms until they are", (5, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.putText(img, "parallel to the ground", (5, 780), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    else:
        cv2.putText(img, "Hold the pose for 10 seconds", (50, 750), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return img


# Initialisation
cap = cv2.VideoCapture(0)
pTime = 0
athlete = pm.poseDetector()

# Real Time caption and correction
while True:
    success, img = cap.read()
    img = cv2.resize(img, (960, 540))
    img = imutils.rotate_bound(img, 90)

    # Show pose to follow
    img[0:200, 0:200] = cv2.imread("Pose/2.jpg")

    # Calculate fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (300, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Detect athlete
    img = athlete.findPose(img)
    # Get landmarks
    lmList = athlete.findPosition(img)

    # Position athlete in the center of the image
    msg = athlete.inPicture(img)
    y0, dy = 800, 50
    for i, line in enumerate(msg.split('\n')):
        y0 = y0 + dy
        cv2.putText(img, line, (10, y0), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


    # if len(lmList)!=0:
    #     print( int(athlete.findAngle(img, 13, 11, 14)))

    if msg == "Follow the pose at the top left":
        warrior(athlete, img)

    cv2.imshow("ATHLETTO", img)
    cv2.waitKey(1)
