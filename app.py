from flask import Flask, render_template, request, redirect
import boto3, os 
from prediction import predict_views_for_new_thumbnail
from rekognition import analyze_thumbnail
import configparser
import base64

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
    if 'thumbnail1' not in request.files or 'thumbnail2' not in request.files:
        return render_template('index.html', error="두 개의 파일을 모두 선택해주세요.")
    
    file1 = request.files['thumbnail1']
    file2 = request.files['thumbnail2']
    
    if file1.filename == '' or file2.filename == '':
        return render_template('index.html', error="두 개의 파일을 모두 선택해주세요.")

    # 첫 번째 이미지 처리
    file1_bytes = file1.read()
    texts1 = analyze_thumbnail(file1_bytes)
    prediction1 = predict_views_for_new_thumbnail(
        'static/thumbnail_model_20241104_075327.joblib',
        'static/tfidf_vectorizer_20241104_075327.joblib', 
        texts1
    )

    # 두 번째 이미지 처리
    file2_bytes = file2.read()
    texts2 = analyze_thumbnail(file2_bytes)
    prediction2 = predict_views_for_new_thumbnail(
        'static/thumbnail_model_20241104_075327.joblib',
        'static/tfidf_vectorizer_20241104_075327.joblib', 
        texts2
    )

    # 더 높은 예측값을 가진 이미지 선택
    if prediction1 >= prediction2:
        best_prediction = prediction1
        best_image_bytes = file1_bytes
    else:
        best_prediction = prediction2
        best_image_bytes = file2_bytes

    # 이미지를 base64로 인코딩
    best_image_b64 = base64.b64encode(best_image_bytes).decode('utf-8')

    return render_template('index.html', prediction=best_prediction, best_image=best_image_b64)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)