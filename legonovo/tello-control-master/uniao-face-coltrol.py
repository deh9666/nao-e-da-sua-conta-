import threading
import socket
import time
from time import sleep

import cv2
import mediapipe as mp
import face_recognition
import pickle   
import numpy as np

# Load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings/data.db", "rb").read())

# host = ''
# port = 9000

# locaddr = (host, port)

# # Create a UDP socket
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# tello_address = ('192.168.10.1', 8889)


# def recv():
#     while True:
#         try:
#             data, server = sock.recvfrom(1518)
#             print(data.decode(encoding="utf-8"))
#         except Exception:
#             print('\nExit . . .\n')
#             break


# Create and start recvThread
# recvThread = threading.Thread(target=recv)
# recvThread.start()

# Initialize Mediapipe models
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands

last_sent = 0
hands_detected = False
hand_results= []
admin_in_the_house = False
read = False
fps = 0
people = []

displacement_x_left = 0
displacement_y_left = 0
displacement_x_right = 0
displacement_y_right = 0

left_hand_center = [0, 0]
right_hand_center = [0, 0]

mutex = threading.Lock()


def detect():
    global hand_results, hands_detected
    global people, admin_in_the_house, frame, read

    # Initialize face and hand detectors
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while True:
        if not read:
            continue

        with mutex:
            image = frame # para garantir que a imagem não seja atualizada durante o processamento

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Initialize the list of names for each face detected
        people = []

        j = 0
        # Loop over the facial embeddings
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            people.append([name, boxes[j]])
            j = j+1

        if admin_in_the_house:
            
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            hand_results = hands_detector.process(rgb_frame)

            if hand_results.multi_hand_landmarks and len(hand_results.multi_hand_landmarks) == 2:
                hands_detected = True
            else:
                hands_detected = False

        time.sleep(0.5)

def control():
    global left_hand_center, right_hand_center, center_admin_face, hands_detected
    global control_started, admin_in_the_house
    global displacement_x_left, displacement_y_left, displacement_x_right, displacement_y_right 

    while True:
        if not hands_detected or not admin_in_the_house:
            displacement_x_left = 0
            displacement_y_left = 0
            displacement_x_right = 0
            displacement_y_right = 0 
            continue

        
        # Calculate hand displacements relative to the face center
        displacement_x_left = center_admin_face[0] - left_hand_center[0]
        displacement_y_left = center_admin_face[1] - left_hand_center[1]
        displacement_x_right = right_hand_center[0] - center_admin_face[0]
        displacement_y_right = center_admin_face[1] - right_hand_center[1]

        # if hands_detected:
        #     # Control drone movement based on hand positions
        #     if displacement_x_left < 60:
        #         print('Movendo o drone 70 cm para a esquerda')
        #         my_drone.move_right(70)

        #     elif displacement_x_left > 150:
        #         print('Movendo o drone 50 cm para a direita')
        #         my_drone.move_left(50)

        #     if displacement_y_right < -180:
        #         print('Move 5cm para a direita')
        #         my_drone.move_forward(50)

        #     elif displacement_y_right > 60:
        #         print('Move 5cm para a esquerda')
        #         my_drone.move_back(70)
        time.sleep(0.05)



def capture():
    global read, frame, fps

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    
    frame_counter = 0
    start_time = time.time()

    while True:
        with mutex:
            read, input = camera.read()
            if not read:
                continue

            frame = cv2.flip(input, 1)


        frame_counter += 1
        fps = frame_counter / (time.time() - start_time)

        sleep(0.01)

# Create and start captureThread
captureThread = threading.Thread(target=capture)
captureThread.start()

# Create and start detectThread
camThread = threading.Thread(target=detect)
camThread.start()

# Create controlThread
controlThread = threading.Thread(target=control)

control_started = False
admin_state = 'Sem admin'
admins = ['deborah', 'douglas']
admin_out = 0

while True:
    if not read:
        continue

    with mutex:
        image = frame


    frame_height, frame_width, _ = image.shape

    for name, (y1, x1, y2, x2) in people:
        if name in admins:
            if not admin_in_the_house: 
                admin_state = 'Admin: ' + name
                admin_in_the_house = True

                if not control_started:
                    controlThread.start()
                    control_started = True
            
            admin_out = 0
            center_admin_face = (int((x1+x2)/2), int((y1+y2)/2))
            # Draw key point on admin face
            cv2.circle(image, (center_admin_face[0], center_admin_face[1]), 5, (255, 0, 0), -1)


        elif admin_in_the_house:
            admin_out = admin_out + 1
            if admin_out >= 30:
                admin_state = 'Sem admin'
                admin_in_the_house = False

        # Desenhando retangulo da face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, name, (x2, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    if len(people) == 0 and admin_in_the_house:
        admin_out = admin_out + 1
        if admin_out >= 30:
            admin_state = 'Sem admin'
            admin_in_the_house = False
       
    if hands_detected:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            hand_points = []

            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                hand_points.append((x, y))
                # cv2.circle(image, (x, y), 5, (255, 0, 255), cv2.FILLED)

            # Calculate left hand center
            if hand_results.multi_handedness[i].classification[0].label == "Left":
                left_hand_center = tuple(np.mean([hand_points[0], hand_points[1], hand_points[2], hand_points[5], hand_points[9], hand_points[13], hand_points[17]] , axis=0).astype(int))
                cv2.circle(image, (left_hand_center[0], left_hand_center[1]), 5, (255, 0, 0), cv2.FILLED)

            # Calculate right hand center
            if hand_results.multi_handedness[i].classification[0].label == "Right":
                right_hand_center = tuple(np.mean([hand_points[0], hand_points[1], hand_points[2], hand_points[5], hand_points[9], hand_points[13], hand_points[17]] , axis=0).astype(int))
                cv2.circle(image, (right_hand_center[0], right_hand_center[1]), 5, (255, 0, 0), cv2.FILLED)

    cv2.putText(image, f"FPS: {fps:.2f} - " + admin_state, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, "Left - x: " + str(displacement_x_left) + " / y: " + str(displacement_y_left), (10, frame_height-40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, "Right - x: " + str(displacement_x_right) + " / y: " + str(displacement_y_right), (10, frame_height-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("frame", image) 
    cv2.waitKey(1)

    sleep(0.001)