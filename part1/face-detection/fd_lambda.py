import json
import base64
import traceback
from io import BytesIO
import torch
import numpy as np
import boto3
from PIL import Image
from facenet_pytorch import MTCNN


AWS_REGION = 'us-east-1'
QUEUE_URL_REQUEST = 'https://sqs.us-east-1.amazonaws.com/<AWS_ACCOUNT_ID>/<STUDENT_ID>-req-queue'


sqs = boto3.client('sqs', region_name=AWS_REGION)


torch.set_grad_enabled(False)
face_detection_model = MTCNN(image_size=240, margin=0, min_face_size=20)


def handler(event, context):
    """Lambda handler for face detection"""
    try:
        # Extract body from event
        body_data = event.get('body', event)
        parsed_body = json.loads(body_data) if isinstance(body_data, str) else body_data
        
        # Get required fields
        encoded_image = parsed_body.get('content')
        request_identifier = parsed_body.get('request_id')
        image_filename = parsed_body.get('filename')
        
        # Validation check
        if not encoded_image or not request_identifier or not image_filename:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing required parameters'})
            }
        
        print(f"Processing request_id: {request_identifier}, filename: {image_filename}")
        
        # Decode and load image
        decoded_bytes = base64.b64decode(encoded_image)
        img_buffer = BytesIO(decoded_bytes)
        pil_image = Image.open(img_buffer).convert("RGB")
        
        # Perform face detection
        try:
            face_result, prob_result = face_detection_model(pil_image, return_prob=True, save_path=None)
        except Exception as detection_error:
            print(f"Error in face detection: {str(detection_error)}")
            traceback.print_exc()
            face_result, prob_result = None, None
        
        # Handle no face detected case
        if face_result is None or prob_result is None:
            print("No face detected")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'request_id': request_identifier,
                    'message': 'No face detected',
                    'filename': image_filename
                })
            }
        
        
        min_val = face_result.min()
        max_val = face_result.max()
        face_normalized = (face_result - min_val) / (max_val - min_val)
        
        # Convert to uint8 image
        face_uint8 = (face_normalized * 255).byte()
        face_hwc = face_uint8.permute(1, 2, 0).numpy()
        
        processed_face = Image.fromarray(face_hwc, mode="RGB")
        
        # Encode to base64
        output_buffer = BytesIO()
        processed_face.save(output_buffer, format="JPEG")
        face_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        # Construct message payload
        queue_message = {
            'request_id': request_identifier,
            'filename': image_filename,
            'face_image': face_b64,
            'probability': float(prob_result)
        }
        
        # Push to SQS queue
        send_result = sqs.send_message(
            QueueUrl=QUEUE_URL_REQUEST,
            MessageBody=json.dumps(queue_message)
        )
        
        message_id = send_result['MessageId']
        print(f"Sent message to SQS: {message_id}")
        
        # Return success response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'request_id': request_identifier,
                'message': 'Face detected and sent for recognition',
                'filename': image_filename,
                'sqs_message_id': message_id
            })
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        traceback.print_exc()
        error_response = {
            'error': str(e),
            'error_type': type(e).__name__
        }
        return {
            'statusCode': 500,
            'body': json.dumps(error_response)
        }
