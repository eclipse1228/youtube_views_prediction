# app.py
from flask import Flask, render_template, request, redirect
import boto3, os 
from prediction import predict_views_for_new_thumbnail
from rekognition import analyze_thumbnail
import configparser

# config
config = configparser.ConfigParser()
config.read('config.config')
# AWS env
client = boto3.client('rekognition')
os.environ['AWS_ACCESS_KEY_ID'] = config['DEFAULT']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['DEFAULT']['AWS_SECRET_ACCESS_KEY']
os.environ['AWS_DEFAULT_REGION'] = config['DEFAULT']['AWS_DEFAULT_REGION']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if 'thumbnail' not in request.files:
        return render_template('index.html', error="파일이 선택되지 않았습니다.")
    
    file = request.files['thumbnail']
    if file.filename == '':
        return render_template('index.html', error="파일이 선택되지 않았습니다.")

    # 파일 처리 및 예측
    file_bytes = file.read()
    texts = analyze_thumbnail(file_bytes)  # rekognition.py의 함수 사용
    prediction = predict_views_for_new_thumbnail('static/thumbnail_model_20241104_075327.joblib','static/tfidf_vectorizer_20241104_075327.joblib', texts)
    print(prediction)
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)