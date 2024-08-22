from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

@app.route('/')
def home():
    return "Hello, this is a simple Flask API!"

@app.route('/api', methods=['POST'])
def my_api():
    data = request.get_json()
    response = {
        'message': 'Received!',
        'data': data
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
