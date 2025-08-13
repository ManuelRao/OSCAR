#include <WiFi.h>
#include <esp_now.h>

typedef struct {
  int throttle;
  int steering;
} ControlPacket;

typedef struct {
  float ax, ay, az;
  float gx, gy, gz;
} TelemetryPacket;

ControlPacket controlData;
TelemetryPacket telemetryData;

// Replace with ESP8266 MAC address
uint8_t carMAC[] = {0x5C, 0xCF, 0x7F, 0xB2, 0xE1, 0xF7};

// Telemetry receive callback
void onTelemetryRecv(const esp_now_recv_info_t *info, const uint8_t *incomingData, int len) {
  if (len == sizeof(TelemetryPacket)) {
    memcpy(&telemetryData, incomingData, sizeof(telemetryData));
    Serial.printf("Telemetry: ax=%.2f ay=%.2f az=%.2f | gx=%.2f gy=%.2f gz=%.2f\n",
                  telemetryData.ax, telemetryData.ay, telemetryData.az,
                  telemetryData.gx, telemetryData.gy, telemetryData.gz);
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init failed");
    return;
  }

  esp_now_register_recv_cb(onTelemetryRecv);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, carMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (!esp_now_is_peer_exist(carMAC)) {
    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
      Serial.println("Failed to add peer");
      return;
    }
  }

  Serial.println("Enter control values: <throttle> <steering>");
  Serial.println("Example: 600 90");
}

void loop() {
  static String input = "";

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (input.length() > 0) {
        processInput(input);
        input = "";
      }
    } else {
      input += c;
    }
  }

  delay(20);  // Small delay to allow ESP-NOW to handle background tasks
}

void processInput(String input) {
  input.trim();
  int spaceIndex = input.indexOf(' ');
  if (spaceIndex == -1) {
    Serial.println("Invalid input. Format: <throttle> <steering>");
    return;
  }

  int throttle = input.substring(0, spaceIndex).toInt();
  int steering = input.substring(spaceIndex + 1).toInt();

  throttle = constrain(throttle, -256, 256);
  steering = constrain(steering, 0, 180);

  controlData.throttle = throttle;
  controlData.steering = steering;

  esp_now_send(carMAC, (uint8_t *)&controlData, sizeof(controlData));
  Serial.printf("Sent: throttle=%d steering=%d\n", throttle, steering);
}
