import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os 
import configparser
import boto3

# config
config = configparser.ConfigParser()
config.read('config.config')
# AWS env
client = boto3.client('rekognition')
os.environ['AWS_ACCESS_KEY_ID'] = config['DEFAULT']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['DEFAULT']['AWS_SECRET_ACCESS_KEY']
os.environ['AWS_DEFAULT_REGION'] = config['DEFAULT']['AWS_DEFAULT_REGION']

def predict_views_for_new_thumbnail(model_filename, vectorizer_filename, new_thumbnail):
    """
    new_thumbnail = {
        'content': '새로운 AI 뉴스',
        'age': 30,
        'size': 0.2,
        'emotions': {
            'ANGRY': False,
            'CALM': True,
            'CONFUSED': False,
            'FEAR': False,
            'HAPPY': False,
            'SAD': False,
            'SURPRISED': False
        },
        'gender': {
            'Female': False,
            'Male': True
        }
    }
    """
    # 1. 텍스트 특성
    vectorizer = joblib.load(vectorizer_filename)  # 직접 load() 대신 joblib.load() 사용
    text_features = vectorizer.transform([new_thumbnail['content']])
    
    # 2. 수치형 특성
    numeric_features = np.array([[
        new_thumbnail.get('age', 0),
        new_thumbnail.get('size', 0)
    ]])
    
    # 3. 감정 특성
    emotion_features = np.array([[
        new_thumbnail['emotions'].get('ANGRY', False),
        new_thumbnail['emotions'].get('CALM', False),
        new_thumbnail['emotions'].get('CONFUSED', False),
        new_thumbnail['emotions'].get('FEAR', False),
        new_thumbnail['emotions'].get('HAPPY', False),
        new_thumbnail['emotions'].get('SAD', False),
        new_thumbnail['emotions'].get('SURPRISED', False)
    ]])
    
    # 4. 성별 특성
    gender_features = np.array([[
        new_thumbnail['gender'].get('Female', False),
        new_thumbnail['gender'].get('Male', False)
    ]])
    
    # 5. 특성 결합
    X_new = np.hstack([
        text_features.toarray(),
        numeric_features,
        emotion_features,
        gender_features
    ])
    
    # 예측
    model = joblib.load(model_filename)
    predicted_views = model.predict(X_new)
    return predicted_views[0]