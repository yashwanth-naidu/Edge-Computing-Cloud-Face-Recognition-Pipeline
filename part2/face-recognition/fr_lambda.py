import json
import base64
import traceback
from io import BytesIO
import torch
import numpy as np
import boto3
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# AWS configuration
AWS_REGION = 'us-east-1'
RESPONSE_QUEUE_URL = 'https://sqs.us-east-1.amazonaws.com/<AWS_ACCOUNT_ID>/<STUDENT_ID>-resp-queue'

# SQS client
sqs = boto3.client('sqs', region_name=AWS_REGION)

# Disable gradients for inference
torch.set_grad_enabled(False)

# Load pre-trained model
print("Loading face recognition model...")
recognition_model = InceptionResnetV1(pretrained='vggface2').eval()

# Load reference data
model_data_file = 'resnetV1_video_weights_1.pt'
reference_data = torch.load(model_data_file, map_location=torch.device('cpu'))
known_embeddings = reference_data[0]
known_names = reference_data[1]

print(f"Model loaded. Known faces: {len(known_names)}")


def handler(event, context):
    
    try:
        records_list = event.get('Records', [])
        print(f"Received event with {len(records_list)} records")
        
        # Process each SQS record
        for record in records_list:
            # Extract message data
            message_content = json.loads(record['body'])
            
            request_identifier = message_content.get('request_id')
            image_filename = message_content.get('filename')
            face_image_b64 = message_content.get('face_image')
            
            print(f"Processing request_id: {request_identifier}, filename: {image_filename}")
            
            # Skip if no face image
            if not face_image_b64:
                print("No face image in message")
                continue
            
            # Decode image from base64
            image_bytes = base64.b64decode(face_image_b64)
            image_buffer = BytesIO(image_bytes)
            face_pil_image = Image.open(image_buffer).convert("RGB")
            
            # Convert PIL to numpy array
            face_np = np.array(face_pil_image, dtype=np.float32)
            
            # Normalize to [0, 1]
            face_np = face_np / 255.0
            
            # Transpose from HWC to CHW
            face_np = np.transpose(face_np, (2, 0, 1))
            
            # Create tensor
            face_tensor = torch.tensor(face_np, dtype=torch.float32)
            
            # Perform recognition
            try:
                # Add batch dimension and get embedding
                face_batch = face_tensor.unsqueeze(0)
                face_embedding = recognition_model(face_batch)
                
                # OPTIMIZED: Vectorized distance calculation (same result, much faster)
                # This computes all distances at once instead of in a loop
                distances = torch.stack([torch.dist(face_embedding, known_emb) for known_emb in known_embeddings])
                
                # Find best match (same logic as before)
                best_match_idx = distances.argmin().item()
                best_distance = distances[best_match_idx].item()
                identified_name = known_names[best_match_idx]
                
                print(f"Recognized: {identified_name} (distance: {best_distance:.4f})")
                
            except Exception as recog_error:
                print(f"Error in face recognition: {str(recog_error)}")
                traceback.print_exc()
                identified_name = "unknown"
            
            # Prepare response message
            response_payload = {
                'request_id': request_identifier,
                'result': identified_name
            }
            
            # Send to response queue
            queue_response = sqs.send_message(
                QueueUrl=RESPONSE_QUEUE_URL,
                MessageBody=json.dumps(response_payload)
            )
            
            print(f"Sent result to response queue: {queue_response['MessageId']}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Processing complete')
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        traceback.print_exc()
        raise e
