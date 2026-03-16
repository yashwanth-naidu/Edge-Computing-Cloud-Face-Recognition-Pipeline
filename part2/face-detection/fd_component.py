import json
import base64
import sys
import traceback
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import threading  

# facenet_pytorch library from local directory
sys.path.insert(0, str(Path(__file__).parent))
from facenet_pytorch import MTCNN

# Import AWS IoT Greengrass IPC libraries
import awsiot.greengrasscoreipc
import awsiot.greengrasscoreipc.client as client
from awsiot.greengrasscoreipc.model import (
    SubscribeToIoTCoreRequest,
    IoTCoreMessage,
    QOS
)

# Setup configuration parameters
STUDENT_ID = "YOUR_STUDENT_ID"
SUBSCRIPTION_TOPIC = f"clients/{STUDENT_ID}-IoTThing"
REQUEST_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/<AWS_ACCOUNT_ID>/<STUDENT_ID>-req-queue"
RESPONSE_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/<AWS_ACCOUNT_ID>/<STUDENT_ID>-resp-queue"
REGION = "us-east-1"

# Initialize MTCNN face detector with specified parameters
face_detector = MTCNN(image_size=240, margin=0, min_face_size=20)

# DEDUPLICATION: Track processed request IDs with thread safety
processed_request_ids = set()
request_lock = threading.Lock()  # Thread lock for safe concurrent access


print(f"Starting Face Detection Component. Topic: {SUBSCRIPTION_TOPIC}")

def create_sqs_client():
    """
    Creates and returns an SQS client for message queue operations.
    Returns None if client creation fails.
    """
    try:
        import boto3
        queue_client = boto3.client('sqs', region_name=REGION)
        print(f"SQS client initialized for {REGION}")
        return queue_client
    except Exception as err:
        print(f"SQS client creation failed: {str(err)}")
        traceback.print_exc()
        return None

class MQTTMessageHandler(client.SubscribeToIoTCoreStreamHandler):
    """Handler class for processing incoming MQTT messages"""
    
    def __init__(self):
        super().__init__()

    def on_stream_event(self, event: IoTCoreMessage) -> None:
        """
        Processes each incoming MQTT message containing video frame data.
        Performs face detection and routes results accordingly.
        """
        try:
            # Decode the incoming MQTT payload
            payload = str(event.message.payload, 'utf-8')
            print(f"Message received on {SUBSCRIPTION_TOPIC}")
            
            # Parse JSON data from payload
            msg_data = json.loads(payload)
            base64_image = msg_data['encoded']
            req_id = msg_data['request_id']
            img_name = msg_data['filename']
            
            print(f"Processing: request_id={req_id}, filename={img_name}")
            
            # THREAD-SAFE DEDUPLICATION CHECK 
            with request_lock:
                if req_id in processed_request_ids:
                    print(f"DUPLICATE request_id={req_id} detected. Skipping.")
                    return
                
                # Add to processed set (still inside lock)
                processed_request_ids.add(req_id)
            # ====================================================
            
            # Decode base64 image to bytes
            img_data = base64.b64decode(base64_image)
            pil_img = Image.open(BytesIO(img_data)).convert("RGB")
            np_img = np.array(pil_img)
            processed_img = Image.fromarray(np_img)
            
            # Run face detection on the image
            detected_face, confidence = face_detector(processed_img, return_prob=True, save_path=None)
            
            if detected_face is not None:
                # Face was detected - process it
                print(f"Face detected in {img_name} (confidence: {confidence})")
                
                # Normalize the detected face tensor
                normalized_face = detected_face - detected_face.min()
                normalized_face = normalized_face / normalized_face.max()
                face_array = (normalized_face * 255).byte().permute(1, 2, 0).numpy()
                face_image = Image.fromarray(face_array)
                
                # Convert face to base64 for transmission
                buffer = BytesIO()
                face_image.save(buffer, format="JPEG")
                encoded_face = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Send detected face to request queue
                queue_client = create_sqs_client()
                if queue_client:
                    request_payload = json.dumps({
                        'request_id': req_id,
                        'filename': img_name,
                        'face_image': encoded_face
                    })
                    
                    send_response = queue_client.send_message(
                        QueueUrl=REQUEST_QUEUE_URL,
                        MessageBody=request_payload
                    )
                    print(f"Face sent to request queue. MessageId: {send_response['MessageId']}")
                else:
                    print("Could not obtain SQS client")
            else:
                # No face detected - send negative response directly
                print(f"No face in {img_name}. Routing to response queue.")
                
                queue_client = create_sqs_client()
                if queue_client:
                    negative_result = {
                        'request_id': req_id,
                        'result': 'No-Face'
                    }
                    
                    send_response = queue_client.send_message(
                        QueueUrl=RESPONSE_QUEUE_URL,
                        MessageBody=json.dumps(negative_result)
                    )
                    print(f"No-Face response queued. MessageId: {send_response['MessageId']}")
                else:
                    print("SQS client unavailable for No-Face response")
                    
        except Exception as err:
            print(f"Error processing message: {str(err)}")
            traceback.print_exc()
    
    def on_stream_error(self, error: Exception) -> bool:
        """Handle stream errors"""
        print(f"Stream error occurred: {error}")
        return True
    
    def on_stream_closed(self) -> None:
        """Handle stream closure"""
        print("MQTT stream closed")

# Main execution block
try:
    # Establish connection to Greengrass IPC
    ipc_connection = awsiot.greengrasscoreipc.connect()
    
    # Configure MQTT subscription request
    subscription_req = SubscribeToIoTCoreRequest()
    subscription_req.topic_name = SUBSCRIPTION_TOPIC
    subscription_req.qos = QOS.AT_LEAST_ONCE
    
    # Create message handler and initiate subscription
    msg_handler = MQTTMessageHandler()
    subscribe_op = ipc_connection.new_subscribe_to_iot_core(msg_handler)
    activation_result = subscribe_op.activate(subscription_req)
    activation_result.result(timeout=10)
    
    print(f"Subscription successful: {SUBSCRIPTION_TOPIC}")
    
    # Keep component running indefinitely
    while True:
        import time
        time.sleep(1)
        
except Exception as err:
    print(f"Main execution error: {str(err)}")
    traceback.print_exc()
