"""
Imports for Web-Viewer
"""
from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import join_room, leave_room, send, SocketIO

"""
Import to load GS_Model from render_wrapper.py
"""
from render_wrapper import GS_Model
import camera_pos_utils as camera
import io
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = "development"
socketio = SocketIO(app)

"""
Available models (scenes) that we can load
"""
idxs = {1, 2, 3}
model_1 = []

@app.route('/', methods=["POST", "GET"])
def home():
    session.clear()
    if request.method == "POST":
        name = request.form.get("name")
        code = request.form.get("code")
        join = request.form.get("join", False) # see if they pressed it

        # Basic Error Checking
        if not name and not code:
            return render_template("home.html", error="Please enter email and model index.", code=code, name=name)
        elif not name:
            return render_template("home.html", error="Please enter email.", code=code, name=name)
        else:
            if not code:
                return render_template("home.html", error="Please enter model index.", code=code, name=name)

        if code == "1":
            R_mat = np.array([[-0.8145390529478596, 0.09889517829114354, 0.5798009170915043],
                              [-0.09778674725285423, 0.98069508920201, -0.16933662945969005],
                              [-0.571807557911322, -0.19462814352607866, -0.7969667511653681]]),
            T_vec = np.array([-2.7518888678267177, 0.5298969558367272, 4.8760898433256425])

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

    # Calculate the new pose

    # Render the new view (img)

    # Emit the image to topic img1
    #socketio.emit("img1", {'image':img_data})


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
    #TODO: make error checking code more robust
    if int(code) not in idxs:
        return

    #TODO: support NeRF studio as well

    R, T = camera.decompose_44(np.array(pose))
    img1 = model_1.render_view(R_mat=R, T_vec=T, img_width=1440, img_height=1920, save=False)
    img_data = io.BytesIO()
    img1.save(img_data, "JPEG")

    # Send an initial (placeholder) image to canvas 1
    socketio.emit("img1", {'image':img_data.getvalue()})

@socketio.on("disconnect")
def disconnect():
    print('disconnected')

if __name__ == '__main__':
    model_1 = GS_Model(model_path="/home/cviss/PycharmProjects/gaussian-splatting/output/1e5592be-5")
    socketio.run(app, debug=False, allow_unsafe_werkzeug=True)
