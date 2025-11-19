import serial
import pygame
import time

# ----- CONFIG -----
SERIAL_PORT = 'COM5'     # Change this to your ESP32 COM port
BAUD_RATE = 115200
SEND_INTERVAL = 0.1      # Seconds between sends

# ----- INIT SERIAL -----
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Error opening serial port: {e}")
    exit()

# ----- INIT JOYSTICK -----
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick found!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Using joystick: {joystick.get_name()}")

# ----- MAIN LOOP -----
try:
    while True:
        pygame.event.pump()

        # Read axes
        x_axis = -joystick.get_axis(0)  # Left-right
        thr = -joystick.get_axis(1)  # Forward-backward (inverted)
        brk = -joystick.get_axis(1)

        # Map values
        steering = int((x_axis + 1) * 68)
        throttle = int(((thr + 1) / 2 * 256))

        # Send over serial
        cmd = f"{throttle} {steering}\n"
        ser.write(cmd.encode())

        print(f"Sent: {cmd.strip()}")

        time.sleep(SEND_INTERVAL)

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    ser.close()
    pygame.quit()
    