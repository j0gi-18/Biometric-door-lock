import paho.mqtt.client as mqtt

#add broker IP address
broker_address = "localhost"

# Define topic
topic = "hello/world"

# Create a client instance
client = mqtt.Client()

def on_message(client, userdata, msg):
  # Print received message
  print(f"Received: {msg.payload.decode('utf-8')}")

# Set callback function
client.on_message = on_message

# Connect to broker
client.connect(broker_address)

# Subscribe to topic
client.subscribe(topic)

# Run forever (until interrupted)
client.loop_forever()
