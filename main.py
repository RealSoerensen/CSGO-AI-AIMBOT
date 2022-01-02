from multiprocessing import Pipe, Process
from time import sleep, time
from os import path, environ
import sys
from configparser import ConfigParser
import win32con
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import win32gui
import win32api
from keyboard import is_pressed
import tensorflow as tf
from mss import mss
import numpy as np
import cv2 as cv

# If csgo not found loop
while True:
    try:
        # Get csgo window
        win32gui.FindWindow(None, 'Counter-Strike: Global Offensive')
        break
    except Exception:
        print("CSGO process is not open, please open CSGO")
        sleep(2)

sct = mss()

path = path.dirname(__file__) + "\\assets\\"

# Get config from asset folder
config = ConfigParser()
config.read(path + 'config.ini')

# Read config
CUDA = config['CONFIG']['CUDA']
FPS_COUNTER = bool(config['CONFIG']['FPS'])
RENDER_SIZE = float(config['CONFIG']['RenderSize'])
TEAM = (config['CONFIG']['Team']).lower()
AIMKEY = (config['CONFIG']['Aimkey']).lower()
CONFIDENCE = float(config['CONFIG']['Confidence'])
VISUALIZATION = bool(config['CONFIG']['Visualization'])

# Find and get rect (x1, y1, x2, y2) of CSGO
def window_size():
    try:
        hwnd = win32gui.FindWindow(None, "Counter-Strike: Global Offensive")
        rect = win32gui.GetWindowRect(hwnd)
        return rect
    except Exception:
        print("Failed to find width and height of CSGO.",
        "Please make sure CSGO is open in windowed or fullscreen",
        "windowed")
        sleep(3)
    finally:
        sys.exit()


width = window_size()[2]
height = window_size()[3]
window = {"top": window_size()[0], "left": window_size()[1], "width": width, "height": height}

# Set AI to utilize either CPU or GPU
environ['CUDA_VISIBLE_DEVICES'] = CUDA

# Get less log to improve performance
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
tf.get_logger().setLevel('ERROR')

# Model preparation
PATH_TO_FROZEN_GRAPH = path + "CSGO_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = path + "CSGO_labelmap.pbtxt"
NUM_CLASSES = 4

# Load a Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Read the Tensorflow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def aimbot(aim_x, aim_y):
    crosshair_x = (width / 2) * aim_x
    crosshair_y = (height / 2) * aim_y
    if is_pressed(AIMKEY):
        # Set cursor to the x, y pos and press mouse1
        win32api.SetCursorPos((crosshair_x, crosshair_y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

def grab_image(p_input):
    while True:
        # Grab screen image
        image = np.asarray(sct.grab(window))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Resize image to 60% of original size. Lower % = worse detection
        image = cv.resize(image, None, fx=RENDER_SIZE, fy=RENDER_SIZE)
        # Put image from pipe
        p_input.send(image)

def get_coords(i , boxes):
    mid_x = (boxes[0][i][1] + boxes[0][i][3])
    mid_y = (boxes[0][i][0] + boxes[0][i][2])
    return mid_x, mid_y

def tf_detecion(p_output, p_input2):
    # Setup FPS counter
    if FPS_COUNTER:
        start_time = time()
        # displays the frame rate every 2 second
        DISPLAY_TIME = 2
        # Set primary FPS to 0
        fps = 0
    # Detection

    with tf.compat.v1.Session(graph=detection_graph.as_default()) as sess:
        while True:
            # Get image from pipe
            image = p_output.recv()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_expanded = np.expand_dims(image, axis=0)

            # Actual detection.
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            array_ch = []
            array_c = []
            array_th = []
            array_t = []
            for i, _ in enumerate(boxes[0]):
                if scores[0][i] > CONFIDENCE:
                    if classes[0][i] == 2:  # ch
                        array_ch.append([get_coords(scores, i, boxes)])

                    if classes[0][i] == 1:  # c
                        array_c.append([get_coords(scores, i, boxes)])

                    if classes[0][i] == 4:  # th
                        array_th.append([get_coords(scores, i, boxes)])

                    if classes[0][i] == 3:  # t
                        array_t.append([get_coords(scores, i, boxes)])

            # Which team to shoot after
            if TEAM == "ct":
                if len(array_ch) > 0:
                    aimbot(array_ch[0][0], array_ch[0][1])
                elif len(array_ch) == 0 and len(array_c) > 0:
                    aimbot(array_c[0][0], array_c[0][1])

            if TEAM == "t":
                if len(array_th) > 0:
                    aimbot(array_th[0][0], array_th[0][1])
                elif len(array_th) == 0 and len(array_t) > 0:
                    aimbot(array_t[0][0], array_t[0][1])

            if FPS_COUNTER:
                # Below we calculate our FPS
                fps += 1
                TIME = time() - start_time
                if TIME >= DISPLAY_TIME:
                    print("FPS: ", fps / TIME)
                    fps = 0
                    start_time = time()

            # Visualization of the results of a detection.
            if VISUALIZATION:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2)
                # Send detection image to pipe2
                p_input2.send(image)

def show_image(p_output2):
    while True:
        image_np = p_output2.recv()
        # Show image with detection
        cv.imshow("Detection window", image_np)
        # Press "q" to quit
        if cv.waitKey(25) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    # Pipes
    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()

    # Creating processes
    p1 = Process(target=grab_image, args=(p_input,))
    p2 = Process(target=tf_detecion, args=(p_output, p_input2,))
    p3 = Process(target=show_image, args=(p_output2,))

    # Starting processes
    p1.start()
    p2.start()
    p3.start()
