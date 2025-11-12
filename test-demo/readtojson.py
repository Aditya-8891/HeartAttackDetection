import serial, json, time

ser = serial.Serial('/dev/ttyUSB0', 9600)
time.sleep(2)

with open('ecg_log.json', 'w') as f:
    while True:
        line = ser.readline().decode().strip()
        entry = {
            "timestamp_ms": int(time.time() * 1000),
            "ecg_value": line
        }
        f.write(json.dumps(entry) + "\n")
        print(entry)

