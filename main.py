"""
Imports for Web-Viewer
"""
from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import join_room, leave_room, send, SocketIO

from render_wrapper import DummyCamera, GS_Model

"""
Import to load GS_Model from render_wrapper.py
"""
import camera_pos_utils as camera
import io
import numpy as np
import torch

# Load environment variables
import os

# Set up logging
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config["SECRET_KEY"] = "development"
socketio = SocketIO(app)

"""
Available models (scenes) that we can load
"""
idxs = {1, 2, 3}
model_1 = []

# Store init values of pose and img_data to reset
init_pose_for_reset = None
init_img_data = None


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
            R_mat = np.array([[-0.70811329, -0.21124761, 0.67375813],
                              [0.16577646, 0.87778949, 0.4494483],
                              [-0.68636268, 0.42995355, -0.58655453]])
            T_vec = np.array([-0.32326042, -3.65895232, 2.27446875])
            #R_mat = np.array([[-0.8210050288356835, -0.17857461458472693, 0.5422746994397166], [0.1249652793099283, 0.8705835196074918, 0.47588655618206266], [-0.5570766747885881, 0.45847076505896, -0.6924378210299766]])
            #T_vec = np.array([-1.8636133065164748, 2.1165406815192687, 3.141789771805336])
            init_pose = camera.compose_44(R_mat, T_vec)
        else:
            init_pose = np.eye(4)

        session["name"] = name
        session["code"] = code
        session["pose"] = init_pose.tolist()

        global init_pose_for_reset
        init_pose_for_reset = init_pose

        return redirect(url_for("viewer"))

    return render_template("home.html")


@app.route("/viewer")
def viewer():
    code = session.get("code")
    if code is None or session.get("name") is None:
        return redirect(url_for("home"))

    return render_template("viewer.html")


@socketio.on('key_control')
def key_control(data):
    """
    key_control listens for button presses from the client...
    1. then calculates a new pose
    2. get new view using gaussian splatting
    3. emit the image to the server

    :param key: keyboard input
    :return:
    """
    key = data.get('key')
    step = data.get('step')
    
    code = session.get("code")
    name = session.get("name")
    pose = session.get("pose")
    pose = np.array(pose)
    logging.info(f'{name} pressed {key} in model {code}, by {step} steps')
    
    """
    SIBR Viewer Controls
    
    Right-Hand Camera Rotations j,l (rot around y), i,k (rotation around x), u,o (rotation around z)
    Left-Hand Camera Translation: a,d (left right), e,f (forward back), q,e (up down)
    We should also handle pressing multiple keys at the same time...
    """
    # Initialize C2C Transform Matricies
    C2C_Rot = np.eye(4, dtype=np.float32)
    C2C_T = np.eye(4, dtype=np.float32)

    # Calculate the new pose
    if key == "d":
        C2C_T = camera.translate4(-0.1 * step, 0, 0)
    elif key == "a":
        C2C_T = camera.translate4(0.1 * step, 0, 0)
    elif key == "s":
        C2C_T = camera.translate4(0, 0, 0.1 * step)
    elif key == "w":
        C2C_T = camera.translate4(0, 0, -0.1 * step)
    elif key == "e":
        C2C_T = camera.translate4(0, 0.1 * step, 0)
    elif key == "q":
        C2C_T = camera.translate4(0, -0.1 * step, 0)
    else:
        C2C_T = np.eye(4, dtype=np.float32)

    if key == "j":
        C2C_Rot = camera.rotate4(np.radians(1 * step), 0, 1, 0)
    elif key == "l":
        C2C_Rot = camera.rotate4(np.radians(-1 * step), 0, 1, 0)
    elif key == "k":
        C2C_Rot = camera.rotate4(np.radians(1 * step), 1, 0, 0)
    elif key == "i":
        C2C_Rot = camera.rotate4(np.radians(-1 * step), 1, 0, 0)
    elif key == "u":
        C2C_Rot = camera.rotate4(np.radians(1 * step), 0, 0, 1)
    elif key == "o":
        C2C_Rot = camera.rotate4(np.radians(-1 * step), 0, 0, 1)
    else:
        C2C_Rot = np.eye(4, dtype=np.float32)

    # Decompose the current pose
    R, T = camera.decompose_44(np.array(pose))
    cam = DummyCamera(R=R, T=T, W=800, H=600, FoVx=1.4261863218, FoVy=1.150908963, C2C_Rot=C2C_Rot, C2C_T=C2C_T)

    img1 = model_1.render_view(cam=cam)

    torch.cuda.empty_cache()  # This should be done periodically... (not everytime)
    img_data = io.BytesIO()  # This is probably also not very efficient
    img1.save(img_data, "JPEG")

    # Emit the image to topic img1
    socketio.emit("img1", {'image': img_data.getvalue()})
    pose = cam.get_new_pose()
    session["pose"] = pose.tolist()  # This might be slow


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

    logger.info(f'connect(): User {name} has requested model {code}.')

    if not code or not name:
        return
    # TODO: make error checking code more robust
    if int(code) not in idxs:
        return

    R, T = camera.decompose_44(np.array(pose))
    cam = DummyCamera(R=R, T=T, W=800, H=600, FoVx=1.4261863218, FoVy=1.150908963)
    img1 = model_1.render_view(cam=cam)
    global init_img_data
    init_img_data = io.BytesIO()
    img1.save(init_img_data, "JPEG")

    # Send an initial (placeholder) image to canvas 1
    socketio.emit("img1", {'image': init_img_data.getvalue()})


@socketio.on("pose_reset")
def image_reset():
    logger.info("Pose reset to initial configuration.")
    
    session["code"] = session.get("code")
    session["name"] = session.get("name")
    global init_pose_for_reset
    session["pose"] = init_pose_for_reset
 
    # Send an initial (placeholder) image to canvas 1
    global init_img_data
    socketio.emit("img1", {'image': init_img_data.getvalue()})
    

@socketio.on("disconnect")
def disconnect():
    name = session.get("name")
    print(f'User {name} has disconnected.')


if __name__ == '__main__':
    config_path = os.getenv('GS_CONFIG_PATH', '/home/cviss/PycharmProjects/GS_Stream/output/dab812a2-1/point_cloud'
                                              '/iteration_30000/config.yaml')
    host = os.getenv('GS_HOST', '127.0.0.1')
    debug = os.getenv('GS_DEBUG', 'false').lower() == 'true'    
    model_1 = GS_Model(config_path=config_path)
    socketio.run(app, host=host, debug=debug, allow_unsafe_werkzeug=True)
