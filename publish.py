import paho.mqtt.client as mqtt
import time

# Define broker address
broker_address = "localhost"

# Define topic
topic = "hello/world"

# Create a client instance
client = mqtt.Client()

#def on_connect(client, userdata, flags, rc):
def on_connect(client, rc):
    # Print connection status whether connected successfully
    if rc == 0:
        print("Connected to broker")
    else:
        print(f"Connection error: {rc}")

# Callback function
client.on_connect = on_connect

# Connect to broker
client.connect(broker_address)

# Continuously publish message until key press
try:
  while True:
    message = "Hello World!"
    client.publish(topic, message)
    time.sleep(1)
except KeyboardInterrupt:
  print("Exiting...")

# Disconnect from broker
client.disconnect()
