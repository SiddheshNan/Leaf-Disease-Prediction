import ast
import json
import socket
import threading
import numpy as np
import imutils
import os
import cv2
from tornado import websocket, web, ioloop, autoreload

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras

lables = open("labels.txt", "r").read()
actions = lables.split("\n")
print("[INFO] loading...")
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('leaf-model.h5', compile=False)
data1 = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


def detect(img):
    # img = cv2.resize(img, (224, 224))
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data1[0] = normalized_image_array
    prediction = model.predict(data1)
    # print(prediction)
    prediction_new = prediction[0].tolist()
    detected_action = prediction_new.index(max(prediction_new))
    print("Detected: " + actions[detected_action])
    detected_acc = max(prediction_new)
    print("Accuracy: " + str(detected_acc))
    return actions[detected_action], str(round(detected_acc, 2))


class CamThread(threading.Thread):
    def __init__(self, ws):
        threading.Thread.__init__(self)
        self.cam = cv2.VideoCapture("http://" + ws.request.remote_ip + ":8080/video")
        self.ws = ws
        self.frame = None
        self.runnable = True

    def run(self):
        while self.runnable:
            ret, self.frame = self.cam.read()
            if self.frame is not None:
                self.frame = imutils.resize(self.frame, width=400)
                cv2.imshow("Frame", self.frame)
                key = cv2.waitKey(1) & 0xFF

                # if the 'q' key is pressed, stop the loop
                if key == ord("q"):
                    self.stop()

    def img_sig(self, message):  # pred here
        print('image signal received')
        if self.frame is not None:
            # frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(self.frame, (224, 224))
            det_action, det_accu = detect(frame)

            cause = "N/A"
            remedie = "N/A"

            if det_action == "0 Cotten-Bacterial":
                cause = "Xanthomonas citri bacteria"
                remedie = "combination of copper and mancozeb-containing fungicides "
            elif det_action == "1 cotton-curlVirus":
                cause = "whitefly Bemisia tabaci"
                remedie = "insecticide treatments against the insect"
            elif det_action == "2 Orange melanose":
                cause = "the plant-pathogenic fungus Diaporthe citri"
                remedie = "use of fungicides"

            json_data = json.dumps({"disease": det_action, "cause": cause, "remedie": remedie})
            self.ws.write_message(json_data)
            # todo: implemnet cause & remedie

    def stop(self):
        self.runnable = False
        self.cam.release()
        cv2.destroyAllWindows()



class SocketHandler(websocket.WebSocketHandler):
    """ Handler for websocket queries. """

    def data_received(self, chunk):
        print(chunk)
        # pass

    def __init__(self, *args, **kwargs):
        self.cam_thread = None
        self.cam = None

        """ Initialize the Redis store and framerate monitor. """
        super(SocketHandler, self).__init__(*args, **kwargs)

    def open(self, *args, **kwargs):
        print('connection opened! client:', self.request.remote_ip)
        self.cam_thread = CamThread(self)
        self.cam_thread.start()

    def on_message(self, message):
        """ Retrieve image ID from database until different from last ID,
        then retrieve image, de-serialize, encode and send to client. """
        j_data = ast.literal_eval(message)
        message = j_data.get('data')
        self.cam_thread.img_sig(message)
        # self.write_message(image)

    def on_close(self):
        self.cam_thread.stop()
        print('closed..')


app = web.Application([
    (r'/ws', SocketHandler),
])

if __name__ == '__main__':
    print([l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1],
                       [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in
                         [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0])
    app.listen(9000)
    autoreload.start()
    print('server started at: http://localhost:9000')
    ioloop.IOLoop.instance().start()
