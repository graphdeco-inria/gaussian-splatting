"""
Imports for Web-Viewer
"""
from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import join_room, leave_room, send, SocketIO
import time

"""
Import to load GS_Model from render_wrapper.py
"""
from render_wrapper import GS_Model
import camera_pos_utils as camera
import io
import numpy as np
import torch

app = Flask(__name__)
app.config["SECRET_KEY"] = "development"
socketio = SocketIO(app)

"""
Available models (scenes) that we can load
"""
idxs = {1, 2, 3}
model_1 = []


M = np.eye(4, dtype=np.float64)
M[1, 1] = -1
M[2, 2] = -1

M_inv = np.linalg.inv(M)

@app.route('/', methods=["POST", "GET"])
def home():
    session.clear()
    if request.method == "POST":
        name = request.form.get("name")
        code = request.form.get("code")
        join = request.form.get("join", False)  # see if they pressed it

        # Basic Error Checking
        if not name and not code:
            return render_template("home.html", error="Please enter email and model index.", code=code, name=name)
        elif not name:
            return render_template("home.html", error="Please enter email.", code=code, name=name)
        else:
            if not code:
                return render_template("home.html", error="Please enter model index.", code=code, name=name)

        if code == "1":
            R_mat = np.array([[-0.8210050288356835, -0.17857461458472693, 0.5422746994397166], [0.1249652793099283, 0.8705835196074918, 0.47588655618206266], [-0.5570766747885881, 0.45847076505896, -0.6924378210299766]])
            T_vec = np.array([-1.8636133065164748, 2.1165406815192687, 3.141789771805336])
            init_pose = camera.compose_44(R_mat, T_vec)
        else:
            init_pose = np.eye(4)

        session["name"] = name
        session["code"] = code
        session["pose"] = init_pose.tolist()
        return redirect(url_for("viewer"))

    return render_template("home.html")


@app.route("/viewer")
def viewer():
    code = session.get("code")
    if code is None or session.get("name") is None:
        return redirect(url_for("home"))

    return render_template("viewer.html")


@socketio.on('key_control')
def key_control(key):
    """
    key_control listens for button presses from the client...
    1. then calculates a new pose
    2. get new view using gaussian splatting
    3. emit the image to the server

    :param key: keyboard input
    :return:
    """
    code = session.get("code")
    name = session.get("name")
    pose = session.get("pose")
    pose = np.array(pose)
    print(f'{name} pressed {key["key"]} in model {code}')

    """
    SIBR Viewer Controls
    
    Right-Hand Camera Rotations j,l (rot around y), i,k (rotation around x), u,o (rotation around z)
    Left-Hand Camera Translation: a,d (left right), e,f (forward back), q,e (up down)
    We should also handle pressing multiple keys at the same time...
    """

    # Calculate the new pose
    if key["key"] == "d":
        pose = camera.translate4(pose, -0.1, 0, 0)
    elif key["key"] == "a":
        pose = camera.translate4(pose, 0.1, 0, 0)
    elif key["key"] == "s":
        pose = camera.translate4(pose, 0, 0, 0.1)
    elif key["key"] == "w":
        pose = camera.translate4(pose, 0, 0, -0.1)
    elif key["key"] == "e":
        pose = camera.translate4(pose, 0, 0.1, 0)
    elif key["key"] == "q":
        pose = camera.translate4(pose, 0, -0.1, 0)
    elif key["key"] == "j":
        pose = camera.rotate4(pose, np.radians(1), 0, 1, 0)
    elif key["key"] == "l":
        pose = camera.rotate4(pose, np.radians(-1), 0, 1, 0)
    elif key["key"] == "k":
        pose = camera.rotate4(pose, np.radians(1), 1, 0, 0)
    elif key["key"] == "i":
        pose = camera.rotate4(pose, np.radians(-1), 1, 0, 0)
    elif key["key"] == "u":
        pose = camera.rotate4(pose, np.radians(1), 0, 0, 1)
    elif key["key"] == "o":
        pose = camera.rotate4(pose, np.radians(-1), 0, 0, 1)
    else:
        pose = pose

    # Render the new view (img)
    #R, T = camera.decompose_44(np.linalg.inv(np.array(pose)))
    R, T = camera.decompose_44(np.array(pose))
    img1 = model_1.render_view(R_mat=R, T_vec=T, img_width=720, img_height=990, save=False)
    torch.cuda.empty_cache()  # This should be done periodically... (not everytime)
    img_data = io.BytesIO()  # This is probably also not very efficient
    img1.save(img_data, "JPEG")

    # Emit the image to topic img1
    socketio.emit("img1", {'image': img_data.getvalue()})
    session["pose"] = pose.tolist()  # This might be slow


"""
handel_message listens for unspecified websocket messages from the client
"""


@socketio.on('my_event')
def handle_message(data):
    print('received message', data)


@socketio.on("connect")
def connect():
    """
    Socket connection event handler
    Gets the code (model index) and name (email) of the requester.
    :return:
    """
    code = session.get("code")
    name = session.get("name")
    pose = session.get("pose")

    print(f'User {name} has requested model {code}.')

    if not code or not name:
        return
    # TODO: make error checking code more robust
    if int(code) not in idxs:
        return

    # TODO: support NeRF studio as well

    R, T = camera.decompose_44(np.array(pose))
    img1 = model_1.render_view(R_mat=R, T_vec=T, img_width=720, img_height=990, save=False)
    img_data = io.BytesIO()
    img1.save(img_data, "JPEG")

    # Send an initial (placeholder) image to canvas 1
    socketio.emit("img1", {'image': img_data.getvalue()})


@socketio.on("disconnect")
def disconnect():
    print('disconnected')


if __name__ == '__main__':
    model_1 = GS_Model(model_path="/home/cviss/PycharmProjects/GS_Stream/output/566ecd8e-c")
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)
