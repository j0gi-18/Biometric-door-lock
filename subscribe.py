import paho.mqtt.client as mqtt

# Define broker address
# broker_address = "192.168.137.209" (opi addrs)
# mac adrs = 192.168.137.233

#broker_address = "192.168.137.86"
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
