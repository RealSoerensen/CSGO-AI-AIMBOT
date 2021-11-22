from win32con import MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP
from win32gui import FindWindow, GetWindowRect
from win32api import SetCursorPos, mouse_event
from multiprocessing import Pipe, Process
from configparser import ConfigParser
from keyboard import is_pressed
from time import sleep, time
from os import path, environ
import tensorflow as tf
from mss import mss
import numpy as np
import cv2 as cv
import sys

# If csgo not found loop
flag = True
while flag:
    try:
        # Get csgo window
        FindWindow(None, 'Counter-Strike: Global Offensive')
        flag = False
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
FPS = bool(config['CONFIG']['FPS'])
renderSize = float(config['CONFIG']['RenderSize'])
team = (config['CONFIG']['Team']).lower()
aimkey = (config['CONFIG']['Aimkey']).lower()
confidence = float(config['CONFIG']['Confidence'])
visualization = bool(config['CONFIG']['Visualization'])

# Setup FPS counter
if FPS:
    startTime = time()
    # displays the frame rate every 2 second
    displayTime = 2
    # Set primary FPS to 0
    fps = 0


# Find and get rect (x1, y1, x2, y2) of CSGO
def windowSize():
    try:
        hwnd = FindWindow(None, "Counter-Strike: Global Offensive")
        rect = GetWindowRect(hwnd)
        return rect
    except Exception:
        print("Failed to find width and height of CSGO.\n Please make sure CSGO is open in windowed or fullscreen "
              "windowed")
        sleep(3)
    finally:
        sys.exit()


width = windowSize()[2]
height = windowSize()[3]
window = {"top": windowSize()[0], "left": windowSize()[1], "width": width, "height": height}

# Set AI to utilize either CPU or GPU
environ['CUDA_VISIBLE_DEVICES'] = CUDA

# Get less log to improve performance
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
tf.get_logger().setLevel('ERROR')

# Env setup
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Model preparation 
PATH_TO_FROZEN_GRAPH = path + "CSGO_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = path + "CSGO_labelmap.pbtxt"
NUM_CLASSES = 4

# Load a Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
categoryIndex = label_map_util.create_category_index(categories)

# Read the Tensorflow model
detectionGraph = tf.Graph()
with detectionGraph.as_default():
    odGraphDef = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serializedGraph = fid.read()
        odGraphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(odGraphDef, name='')


def aimbot(aim_x, aim_y):
    crosshair_x = (width / 2) * aim_x
    crosshair_y = (height / 2) * aim_y
    if is_pressed(aimkey):
        # Set cursor to the x, y pos and press mouse1
        SetCursorPos((crosshair_x, crosshair_y))
        mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0)
        mouse_event(MOUSEEVENTF_LEFTUP, 0, 0)


def grabImage(p_input):
    while True:
        # Grab screen image
        image = np.asarray(sct.grab(window))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Resize image to 60% of original size. Lower % = worse detection
        image = cv.resize(image, None, fx=renderSize, fy=renderSize)
        # Put image from pipe
        p_input.send(image)


def TensorflowDetection(p_output, p_input2):
    global startTime, fps
    # Detection
    with detectionGraph.as_default():
        with tf.compat.v1.Session(graph=detectionGraph) as sess:
            while True:
                # Get image from pipe
                image = p_output.recv()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                imageExpanded = np.expand_dims(image, axis=0)

                # Actual detection.
                image_tensor = detectionGraph.get_tensor_by_name('image_tensor:0')
                boxes = detectionGraph.get_tensor_by_name('detection_boxes:0')
                scores = detectionGraph.get_tensor_by_name('detection_scores:0')
                classes = detectionGraph.get_tensor_by_name('detection_classes:0')
                num_detections = detectionGraph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: imageExpanded})

                array_ch = []
                array_c = []
                array_th = []
                array_t = []
                for i, _ in enumerate(boxes[0]):
                    if classes[0][i] == 2:  # ch
                        if scores[0][i] >= confidence:
                            mid_x = (boxes[0][i][1] + boxes[0][i][3])
                            mid_y = (boxes[0][i][0] + boxes[0][i][2])
                            array_ch.append([mid_x, mid_y])
                            cv.circle(image, (int(mid_x * width), int(mid_y * height)), 3, (0, 0, 255), -1)

                    if classes[0][i] == 1:  # c
                        if scores[0][i] >= confidence:
                            mid_x = (boxes[0][i][1] + boxes[0][i][3])
                            mid_y = boxes[0][i][0] + (boxes[0][i][2] - boxes[0][i][0])
                            array_c.append([mid_x, mid_y])
                            cv.circle(image, (int(mid_x * width), int(mid_y * height)), 3, (50, 150, 255), -1)

                    if classes[0][i] == 4:  # th
                        if scores[0][i] >= confidence:
                            mid_x = (boxes[0][i][1] + boxes[0][i][3])
                            mid_y = (boxes[0][i][0] + boxes[0][i][2])
                            array_th.append([mid_x, mid_y])
                            cv.circle(image, (int(mid_x * width), int(mid_y * height)), 3, (0, 0, 255), -1)

                    if classes[0][i] == 3:  # t
                        if scores[0][i] >= confidence:
                            mid_x = (boxes[0][i][1] + boxes[0][i][3])
                            mid_y = boxes[0][i][0] + (boxes[0][i][2] - boxes[0][i][0])
                            array_t.append([mid_x, mid_y])
                            cv.circle(image, (int(mid_x * width), int(mid_y * height)), 3, (50, 150, 255), -1)

                # Which team to shoot after
                if team == "ct":
                    if len(array_ch) > 0:
                        aimbot(array_ch[0][0], array_ch[0][1])
                    elif len(array_ch) == 0 and len(array_c) > 0:
                        aimbot(array_c[0][0], array_c[0][1])

                if team == "t":
                    if len(array_th) > 0:
                        aimbot(array_th[0][0], array_th[0][1])
                    elif len(array_th) == 0 and len(array_t) > 0:
                        aimbot(array_t[0][0], array_t[0][1])

                if FPS:
                    # Below we calculate our FPS
                    fps += 1
                    TIME = time() - startTime
                    if TIME >= displayTime:
                        print("FPS: ", fps / TIME)
                        fps = 0
                        startTime = time()

                # Visualization of the results of a detection.
                if visualization:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        categoryIndex,
                        use_normalized_coordinates=True,
                        line_thickness=2)
                    # Send detection image to pipe2
                    p_input2.send(image)


def showImage(p_output2):
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
    p1 = Process(target=grabImage, args=(p_input,))
    p2 = Process(target=TensorflowDetection, args=(p_output, p_input2,))
    p3 = Process(target=showImage, args=(p_output2,))

    # Starting processes
    p1.start()
    p2.start()
    p3.start()
