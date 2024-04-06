import Jetson.GPIO as GPIO
import time

servoPIN = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
p.start(2.5) # Initialization

class servo:
    def rotate(self, angle):
        p.ChangeDutyCycle((angle/18)+2.5)
    def __del__(self):
        GPIO.cleanup()
#try:
#  while True:
#    for x in range(0,180):
#        p.ChangeDutyCycle(get_pwm(x))
#        print(x)
#        time.sleep(0.2)
#    for x in range(180,0,-1):
#        p.ChangeDutyCycle(get_pwm(x))
#        #print(50)
#        time.sleep(0.2)
#except KeyboardInterrupt:
#  p.stop()
#  GPIO.cleanup()
