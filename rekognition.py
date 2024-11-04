import boto3
import pandas as pd

client = boto3.client('rekognition')

def analyze_thumbnail(file):
    thumbnail_texts = {
        'content': '',
        'age': 0,
        'size': 0,
        'emotions': {
            'ANGRY': False,
            'CALM': False,
            'CONFUSED': False,
            'FEAR': False,
            'HAPPY': False,
            'SAD': False,
            'SURPRISED': False
        },
        'gender': {
            'Female': False,
            'Male': False
        }
    }
    response = client.detect_text(Image={'Bytes': file})
    for text in response['TextDetections']:
        if text['Confidence'] > 70:
            thumbnail_texts['content'] = text['DetectedText']
    
    response = client.detect_faces(Image={'Bytes': file},Attributes=['ALL'])
    max_height = 0
    for faceDetail in response['FaceDetails']:
        # size
        height = faceDetail['BoundingBox']['Height']
        
        if height > max_height:
            max_height = height
        else: 
            continue
        
        width = faceDetail['BoundingBox']['Width']
        
        # age는 예상 나이 범위의 (low+high)/2
        age = (faceDetail['AgeRange']['Low'] + faceDetail['AgeRange']['High']) / 2
        
        # size
        thumbnail_texts['size'] = width * height
        
        # emotion
        for emotion in faceDetail['Emotions']:
            if emotion['Confidence'] > 70:
                match emotion['Type']:
                    case 'ANGRY':
                        thumbnail_texts['emotions']['ANGRY'] = True
                    case 'CALM':
                        thumbnail_texts['emotions']['CALM'] = True
                    case 'CONFUSED':
                        thumbnail_texts['emotions']['CONFUSED'] = True
                    case 'FEAR':
                        thumbnail_texts['emotions']['FEAR'] = True
                    case 'HAPPY':
                        thumbnail_texts['emotions']['HAPPY'] = True
                    case 'SAD':
                        thumbnail_texts['emotions']['SAD'] = True
                    case 'SURPRISED':
                        thumbnail_texts['emotions']['SURPRISED'] = True
        
        # gender
        if faceDetail['Gender']['Confidence'] > 70:
            if faceDetail['Gender']['Value'] == 'Female':
                thumbnail_texts['gender']['Female'] = True
            else:
                thumbnail_texts['gender']['Male'] = True
        
    return thumbnail_texts

