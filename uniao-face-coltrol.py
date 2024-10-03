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


import rpyc

# Create a RPyC connection to the remote ev3dev device.
# Use the hostname or IP address of the ev3dev device.
# If this fails, verify your IP connectivty via ``ping X.X.X.X``
#conn = rpyc.classic.connect('198.168.1.125')

# conn = rpyc.classic.connect('192.168.0.125', port=18812)
conn = rpyc.classic.connect('192.168.0.125')



# import ev3dev2 on the remote ev3dev device
ev3dev2_motor = conn.modules['ev3dev2.motor']
ev3dev2_sensor = conn.modules['ev3dev2.sensor']
ev3dev2_sensor_lego = conn.modules['ev3dev2.sensor.lego']


# Usar LargeMotor e TouchSensor no dispositivo remoto ev3dev
# motor_left = ev3dev2_motor.LargeMotor(ev3dev2_motor.OUTPUT_A)
# motor_right = ev3dev2_motor.LargeMotor(ev3dev2_motor.OUTPUT_B)

# Use LargeMotor on the remote ev3dev device
motor_esquerdo = ev3dev2_motor.LargeMotor(ev3dev2_motor.OUTPUT_A)
motor_direito = ev3dev2_motor.LargeMotor(ev3dev2_motor.OUTPUT_B)
# ts = ev3dev2_sensor_lego.TouchSensor(ev3dev2_sensor.INPUT_1)



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
        
# Função para girar o robô no sentido horário
def girar_horario():
    motor_esquerdo.run_forever(speed_sp=200)
    motor_direito.run_forever(speed_sp=-200)

# Função para girar o robô no sentido anti-horário
def girar_antihorario():
    motor_esquerdo.run_forever(speed_sp=-200)
    motor_direito.run_forever(speed_sp=200)

# Função para mover o robô para trás
def andar_para_tras():
    motor_esquerdo.run_forever(speed_sp=-200)
    motor_direito.run_forever(speed_sp=-200)

def andar_para_frente():
    motor_esquerdo.run_forever(speed_sp=200)
    motor_direito.run_forever(speed_sp=200)

# Função para parar os motores
def parar_motores():
    motor_esquerdo.stop(stop_action="brake")
    motor_direito.stop(stop_action="brake")


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

            # motor_left.run_forever(speed_sp=500)
            # motor_right.run_forever(speed_sp=500)    

            # motor_left.stop()
            # motor_right.stop()
    
        # Calculate hand displacements relative to the face center
        displacement_x_left = center_admin_face[0] - left_hand_center[0]
        # print("deslocamento esquerdo",displacement_x_left)
        displacement_y_left = center_admin_face[1] - left_hand_center[1]
        # print("deslocamento esquerda",displacement_x_left)
        displacement_x_right = right_hand_center[0] - center_admin_face[0]
        # print("deslocamento direita",displacement_y_right)
        displacement_y_right = center_admin_face[1] - right_hand_center[1]

        if hands_detected:
            # Control drone movement based on hand positions
            if displacement_x_left < 90:
                print('Movendo para frente')
                andar_para_frente()
                
            elif 90 <= displacement_x_left < 180:
                parar_motores()

            elif displacement_x_left > 180:
                print('Movendo para a trás')
                andar_para_tras()
                
            

            if displacement_y_right < -180:
                print('girar o robô no sentido horário')
                girar_horario()
                

            elif -180 <= displacement_y_right <= 80:
                parar_motores()


            elif displacement_y_right > 80:
                print('girar o robô no sentido anti-horário')
                girar_antihorario()
                
                
                
        time.sleep(0.1)



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

    # sleep(0.001)