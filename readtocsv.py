import serial, csv, time

ser = serial.Serial('/dev/ttyUSB0', 9600)  # Change port name (e.g., COM3 on Windows)
time.sleep(2)  # Give time for Arduino to reset

with open('ecg_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp(ms)', 'ecg_value'])
    
    while True:
        line = ser.readline().decode().strip()
        writer.writerow([int(time.time() * 1000), line])
        print(line)
        f.flush()  # Ensure data is written to file immediately

