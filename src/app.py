import base64
import json

from io import BytesIO
from PIL import Image

from image_classifier import ImageClassifier

# create an Image Classifier instance
image_classifier = ImageClassifier()
        
def lambda_handler(event, context):
    if event['isBase64Encoded'] is not True:
        return {
            'statusCode': 400 # Bad Request!
        }
    
    
    # The event body will have a base64 string containing the image bytes,
    # Deocde the base64 string into bytes
    image_bytes = base64.b64decode(event['body'])

    # Create a Pillow Image from the image bytes
    image = Image.open(BytesIO(image_bytes))
    
    # Use the image classifier to get a prediction for the image
    value, score = image_classifier.classify(image)
    
    # return the response as JSON
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
        },
        "body": json.dumps({
            'value': value,
            'score': score 
        })
    }
