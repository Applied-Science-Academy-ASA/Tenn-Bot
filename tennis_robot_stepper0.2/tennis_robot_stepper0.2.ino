TaskHandle_t Task1;
TaskHandle_t Task2;

#define dirPin 2
#define stepPin 4
#define servoPin 13
#define servoPin2 14
#define Relay 12

#include <ESP32_Servo.h>
Servo myservo;
Servo myservo2;

#include <AccelStepper.h>
AccelStepper stepper; // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5

String commandstr = "";
int SetMaxSpeed = 3200;
int SetAcceleration = 3200;
bool startcon[2] = {false, false};
int pos = 0;
int pos2 = 0;
int command = 320;

void setup() {
  pinMode (stepPin, OUTPUT);
  pinMode (dirPin, OUTPUT);
  pinMode (Relay, OUTPUT);
  Serial.begin(115200);
  Serial.setTimeout(0.01);

  myservo.attach(servoPin);
  myservo2.attach(servoPin2);
  stepper.setMaxSpeed(SetMaxSpeed);
  stepper.setSpeed(SetMaxSpeed);
  stepper.setAcceleration(SetAcceleration);

  //  xTaskCreatePinnedToCore(
  //    TH1,           //func
  //    "Task1",      //name anything
  //    10000,        //stack size
  //    NULL,         //param
  //    1,            //priority
  //    &Task1,       //Handler
  //    0             //pin to core 0
  //  );
  //
  //  xTaskCreatePinnedToCore(
  //    TH2,           //func
  //    "Task2",      //name anything
  //    10000,        //stack size
  //    NULL,         //param
  //    1,            //priority
  //    &Task1,       //Handler
  //    1             //pin to core 0
  //  );
}

void TH1() {
  while (true) {
    if (Serial.available()) {
      commandstr = Serial.readString();
      Serial.println(commandstr);
      for (int x = 0; x < commandstr.length() ; x++) {
        if (commandstr[x] == 'l') {
          startcon[0] = true;
        }
        if (commandstr[x] == 'k') {
          startcon[1] = true;
        }
      }
      if (!startcon[0] || !startcon[1]) {
        command = commandstr.toInt();
        if (command > 1000) {
          command = 320;
        }
      }
      Serial.println(command);
      break;
    }
  }
  if (startcon[0]) {
    startcon[0] = false;
    stepper.stop();
    digitalWrite (Relay, HIGH);
    for (pos2 = 30; pos2 <= 45; pos2 += 1) { // goes from 0 degrees to 180 degrees
      // in steps of 1 degree
      myservo2.write(pos2);              // tell servo to go to position in variable 'pos'
      delay(15);                       // waits 15ms for the servo to reach the position
    }
    for (pos2 = 45; pos2 >= 30; pos2 -= 1) { // goes from 180 degrees to 0 degrees
      myservo2.write(pos2);              // tell servo to go to position in variable 'pos'
      delay(15);                       // waits 15ms for the servo to reach the position
    }
    delay(2000);
    for (pos = 30; pos <= 45; pos += 1) { // goes from 0 degrees to 180 degrees
      // in steps of 1 degree
      myservo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);                       // waits 15ms for the servo to reach the position
    }
    for (pos = 45; pos >= 30; pos -= 1) { // goes from 180 degrees to 0 degrees
      myservo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);                       // waits 15ms for the servo to reach the position
    }
    delay(3000);
    digitalWrite (Relay, LOW);
    delay(1000);
    command = 320;
  }
  else if (startcon[1]) {
    startcon[1] = false;
    stepper.stop();
    digitalWrite (Relay, HIGH);
    delay(9000);
    for (pos = 30; pos <= 45; pos += 1) { // goes from 0 degrees to 180 degrees
      // in steps of 1 degree
      myservo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);                       // waits 15ms for the servo to reach the position
    }
    for (pos = 45; pos >= 30; pos -= 1) { // goes from 180 degrees to 0 degrees
      myservo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);                       // waits 15ms for the servo to reach the position
    }
    delay(5000);
    for (pos = 30; pos <= 45; pos += 1) { // goes from 0 degrees to 180 degrees
      // in steps of 1 degree
      myservo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);                       // waits 15ms for the servo to reach the position
    }
    for (pos = 45; pos >= 30; pos -= 1) { // goes from 180 degrees to 0 degrees
      myservo.write(pos);              // tell servo to go to position in variable 'pos'
      delay(15);                       // waits 15ms for the servo to reach the position
    }
    delay(3000);
    digitalWrite (Relay, LOW);
    delay(1000);
    command = 320;
  }
  else if (command < 300) {
    stepper.setCurrentPosition((320 - command) * 6);
    stepper.moveTo(0);

    while (stepper.currentPosition() > 0) {
      stepper.run();
    }
    delay(2000);
  }
  else if (command > 340) {
    stepper.setCurrentPosition(0);
    stepper.moveTo((command - 320) * 6);
    while (stepper.currentPosition() < (command - 320) * 6) {
      stepper.run();
    }
    delay(2000);
  }
  //  else if (command > 0){
  //    stepper.setCurrentPosition(command*8);
  //  }
}

//void TH2(void *param) {
//  if (Serial.available()) {
//    command = Serial.read();
//    Serial.println(command);
//  }
//}

void loop() {
  TH1();
}
