from flask import Flask
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS, cross_origin
from chatbot import ask


app = Flask(__name__)
CORS(app)

# create a Socket.IO server
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@socketio.on('client message event')
def handle_client_message_event(data):
    print('received client message: ' + data['data'])
    response = ask(data['data'])
    emit('server response', {'data': response})


@socketio.on('message')
def handle_message(data):
    print('received message: ' + data['data'])
    emit('server response', {'data': 'got it!'})

#
# @socketio.on('connect')
# def test_connect():
#     print('Client connected')
#     emit('server response', {'data': 'Connected'})

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

# @socketio.on('my broadcast event')
# def test_message(message):
#     print('test test')
#     emit('server response', {'data': message['data']}, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, port=8000)
