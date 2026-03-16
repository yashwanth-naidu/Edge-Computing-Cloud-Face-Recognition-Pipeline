# Edge Computing - Cloud Face Recognition Pipeline

A distributed, real-time face recognition system built on AWS combining 
edge computing with serverless cloud functions to process IoT video frames 
with low latency and minimal cloud overhead.

---

## Architecture Overview

### Part I - Fully Serverless Pipeline (AWS Lambda)
Video frames are sent directly from a client to a Lambda function via HTTP POST.
Face detection and recognition both run in the cloud.

### Part II - Edge + Cloud Hybrid Pipeline (AWS IoT Greengrass)
Face detection runs on the edge device (Greengrass Core), reducing latency 
and cloud usage. Only detected faces are forwarded to the cloud for recognition.

---

## Tech Stack

- AWS Lambda - Serverless face recognition
- AWS IoT Greengrass v2 - Edge face detection component
- AWS SQS - Decoupled message passing between edge and cloud
- AWS ECR - Docker image registry for Lambda functions
- MTCNN - Multi-task Cascaded CNN for face detection
- FaceNet (InceptionResnetV1) - Deep learning model for face recognition
- PyTorch (CPU-optimized) - ML inference framework
- Python 3 - Core implementation language

---

## Project Structure

part1/                          
    face-detection/
        fd_lambda.py            # MTCNN face detection Lambda (HTTP trigger)
    face-recognition/
        fr_lambda.py            # FaceNet recognition Lambda (SQS trigger)

part2/                          
    face-detection/
        fd_component.py         # IoT Greengrass edge component (MQTT subscriber)
    face-recognition/
        fr_lambda.py            # FaceNet recognition Lambda (SQS trigger)

---

## Key Features

- Edge face detection using MTCNN on AWS IoT Greengrass
- Frames with no face handled entirely on the edge, pushing No-Face result
  directly to response queue, skipping cloud invocation entirely
- Serverless recognition using FaceNet embeddings on AWS Lambda via SQS trigger
- Thread-safe deduplication in the Greengrass component for duplicate MQTT messages
- Vectorized embedding comparison in Part II for faster face matching
- Achieved under 1.5s average latency across 100 concurrent requests (Part II)
- Achieved under 3s average latency across 100 concurrent requests (Part I)

---

## Configuration

Queue URLs and AWS account details are configured at deployment time.
Replace AWS_ACCOUNT_ID and STUDENT_ID placeholders in source files before deploying.
