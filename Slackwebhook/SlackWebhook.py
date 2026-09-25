import requests
import time

# Replace with your actual webhook URL
WEBHOOK_URL = ""

def send_slack_message(message):
    payload = {"text": message}
    requests.post(WEBHOOK_URL, json=payload)

# Example: Simulate training alerts
for epoch in range(1, 6):
    accuracy = 0.7 + epoch * 0.05
    msg = f"Training Update - Epoch {epoch}: Accuracy = {accuracy:.2f}"
    send_slack_message(msg)
    time.sleep(2)  # simulate processing time

send_slack_message("✅ Training Completed!")
