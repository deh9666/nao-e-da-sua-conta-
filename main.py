
import rpyc

# Create a RPyC connection to the remote ev3dev device.
# Use the hostname or IP address of the ev3dev device.
# If this fails, verify your IP connectivty via ``ping X.X.X.X``
#conn = rpyc.classic.connect('198.168.1.125')

import rpyc

# Conexão RPyC com o dispositivo ev3dev remoto
conn = rpyc.classic.connect('192.168.0.125')

# Importar ev3dev2 no dispositivo remoto ev3dev
ev3dev2_motor = conn.modules['ev3dev2.motor']
ev3dev2_sensor = conn.modules['ev3dev2.sensor']
ev3dev2_sensor_lego = conn.modules['ev3dev2.sensor.lego']

# Usar LargeMotor e TouchSensor no dispositivo remoto ev3dev
motor_left = ev3dev2_motor.LargeMotor(ev3dev2_motor.OUTPUT_A)
motor_right = ev3dev2_motor.LargeMotor(ev3dev2_motor.OUTPUT_B)
ts = ev3dev2_sensor_lego.TouchSensor(ev3dev2_sensor.INPUT_1)

# Se o TouchSensor for pressionado, o robô vai para frente
while True:
    if ts.is_pressed:
        motor_left.run_forever(speed_sp=500)
        motor_right.run_forever(speed_sp=500)
    else:
        motor_left.stop()
        motor_right.stop()

# conn = rpyc.classic.connect('192.168.0.125')



# # import ev3dev2 on the remote ev3dev device
# ev3dev2_motor = conn.modules['ev3dev2.motor']
# ev3dev2_sensor = conn.modules['ev3dev2.sensor']
# ev3dev2_sensor_lego = conn.modules['ev3dev2.sensor.lego']

# # Use the LargeMotor and TouchSensor on the remote ev3dev device
# motor = ev3dev2_motor.LargeMotor(ev3dev2_motor.OUTPUT_A)
# ts = ev3dev2_sensor_lego.TouchSensor(ev3dev2_sensor.INPUT_1)

# # If the TouchSensor is pressed, run the motor
# while True:
#     ts.wait_for_pressed()
#     motor.run_forever(speed_sp=200)

#     ts.wait_for_released()
#     motor.stop()




#!/usr/bin/env python3

# #import log function
# from ev3devlogging import timedlog as log
# #import ev3 API
# from ev3dev2 import auto as ev3


# m = ev3.LargeMotor(ev3.OUTPUT_A)
# m.on_for_rotations(SpeedPercent(75), 5)

# # initialize
# # ------------
# # initialize left and right motor as tank combo
# tankDrive = ev3.MoveTank(ev3.OUTPUT_A, ev3.OUTPUT_B)

# # initialize some constants
# SPEED_FORWARD = ev3.SpeedPercent(30)     # set speed to 30% of maximum speed
# SPEED_BACKWARD = ev3.SpeedPercent(-30)   # backward with same speed as forward
# SPEED_ZERO = ev3.SpeedPercent(0)       # stop motor (speed is zero)

# TURN_TIME = 0.62

# # main loop
# # -----------

# log("drive forward")
# tankDrive.on_for_seconds(SPEED_FORWARD, SPEED_FORWARD, 2)

# log("turn right")
# tankDrive.on_for_seconds(SPEED_FORWARD, SPEED_BACKWARD, TURN_TIME)

# log("drive forward")
# tankDrive.on_for_seconds(SPEED_FORWARD, SPEED_FORWARD, 3)

# log("turn right")
# tankDrive.on_for_seconds(SPEED_FORWARD, SPEED_BACKWARD, TURN_TIME)

# log("drive forward")
# tankDrive.on_for_seconds(SPEED_FORWARD, SPEED_FORWARD, 2)

# log("turn right")
# tankDrive.on_for_seconds(SPEED_FORWARD, SPEED_BACKWARD, TURN_TIME)

# log("drive forward")
# tankDrive.on_for_seconds(SPEED_FORWARD, SPEED_FORWARD, 3)

# log("turn right")
# tankDrive.on_for_seconds(SPEED_FORWARD, SPEED_BACKWARD, TURN_TIME)

# log("finished")



# #!/usr/bin/env pybricks-micropython
# from pybricks.hubs import EV3Brick
# from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
#                                  InfraredSensor, UltrasonicSensor, GyroSensor)
# from pybricks.parameters import Port, Stop, Direction, Button, Color
# from pybricks.tools import wait, StopWatch, DataLog
# from pybricks.robotics import DriveBase
# from pybricks.media.ev3dev import SoundFile, ImageFile


# # This program requires LEGO EV3 MicroPython v2.0 or higher.
# # Click "Open user guide" on the EV3 extension tab for more information.

# # Create your objects here. 
# ev3 = EV3Brick()
# left_motor = Motor(Port.A)
# right_motor = Motor(Port.B)
# robot = DriveBase(left_motor, right_motor, wheel_diameter=55.5, axle_track = 104)

# # Write your program here.
# # ev3.speaker.beep()
# while True:
#     if Button.CENTER in ev3.buttons.pressed():
#         robot.turn(120)
#         robot.straight(300) 
#         robot.turn(100)
        
#         robot.stop()
