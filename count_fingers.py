import cv2
#ta1
import mediapipe as mp

cap = cv2.VideoCapture(0)

#ta2
mp_hands =mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#ta3
hands =mp_hands.Hands(min_detection_confidence =0.8, min_tracking_confidence = 0.5)

#ta6(define a function to define connections)
def drawHandLandmarks(image, hand_landmarks):
    #ta7(draw connection between landmark points)
    if hand_landmarks:

        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image,landmarks,mp_hands.HAND_CONNECTIONS)


while True:
    success, image = cap.read()

    #ta5
    image= cv2.flip(image,1)

    #ta4(detect the hands Landmark)
    results = hands.process(image)

    #ta7(get landmark image from processed image)
    hand_landmarks = results.multi_hand_landmarks

    #ta8(draw landmarks)
    drawHandLandmarks(image, hand_landmarks)

    cv2.imshow("Media Controller", image)

    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()

