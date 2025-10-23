#include <ESP8266WiFi.h>
#include <espnow.h>
#include <Servo.h>

// --------------------- CONTROL & TELEMETRY STRUCTS ---------------------
typedef struct {
  int throttle;
  int steering;
} ControlPacket;

typedef struct {
  float ax, ay, az;
  float gx, gy, gz;
} TelemetryPacket;

ControlPacket controlData;
uint8_t controllerMAC[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00}; // Replace with your ESP32 MAC

// --------------------- HARDWARE SETUP ---------------------
// Motor A pins
const int ENA = D2; // PWM pin for Motor A
const int IN1 = D0;
const int IN2 = D3;

// Motor B pins
const int ENB = D7; // PWM pin for Motor B
const int IN3 = D5;
const int IN4 = D6;

// Servo pin
const int SERVO_PIN = D8;
Servo steeringServo;

// Servo positions
int leftMax = 10;
int rightMax = 120;
int centerPos = 68;

// --------------------- SETUP ---------------------
void setup() {
  Serial.begin(115200);
  
  // WiFi & ESP-NOW init
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  if (esp_now_init() != 0) {
    Serial.println("ESP-NOW init failed!");
    return;
  }

  esp_now_set_self_role(ESP_NOW_ROLE_COMBO);
  esp_now_register_recv_cb(onDataRecv);
  esp_now_add_peer(controllerMAC, ESP_NOW_ROLE_COMBO, 1, NULL, 0);

  Serial.println(WiFi.macAddress());

  // Motor pins
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // Servo
  steeringServo.attach(SERVO_PIN);
  steeringServo.write(centerPos);

  Serial.println("Car ready. Waiting for commands...");
}

// --------------------- LOOP ---------------------
void loop() {
  applyControl();

  // Fake telemetry (all zeroes)
  TelemetryPacket telemetry = {0, 0, 0, 0, 0, 0};
  esp_now_send(controllerMAC, (uint8_t *)&telemetry, sizeof(telemetry));
  delay(100);  // Adjust telemetry rate as needed
}

// --------------------- CONTROL ---------------------
void onDataRecv(uint8_t *mac, uint8_t *incomingData, uint8_t len) {
  memcpy(&controlData, incomingData, sizeof(controlData));
  Serial.printf("Received: throttle=%d steering=%d\n", controlData.throttle, controlData.steering);
}

void applyControl() {
  int throttle = controlData.throttle;
  int steering = controlData.steering;

  // Motor A
  if(throttle > 0){
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  }else{
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }
  analogWrite(ENA, throttle);
  analogWrite(ENB, throttle);

  

  // Servo
  steeringServo.write(constrain(steering, leftMax, rightMax));
}

//stearing automatic control 