"""
Serve webcam images from a Redis store using Tornado.
Usage:
   python server.py
"""
import ast
import base64
import json
import pickle
import socket
import sys
import threading

import imutils
import os
import cv2
import numpy
from tornado import websocket, web, ioloop, autoreload

MAX_FPS = 100


class CamThread(threading.Thread):
    def __init__(self, ws, model, names, data):
        threading.Thread.__init__(self)
        self.cam = cv2.VideoCapture("http://" + ws.request.remote_ip + ":8080/video")
        self.ws = ws
        self.frame = None
        self.model = model
        self.data = data
        self.runnable = True
        self.names = names

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

    def img_sig(self, message):
        global data
        print('img_sig:' + str(message))
        if self.frame is not None:
            gray_en = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            prediction = self.model.predict(gray_en)
            print(prediction)
            if prediction[1] < 110:
                nam = self.names[prediction[0]]
                cv2.putText(self.frame, nam, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
                # print self.data
                # data = ast.literal_eval(data)
                ls = self.data[nam]
                data2 = json.dumps({"disease": nam, "cause": ls[1], "remedie": ls[0]})
                self.ws.write_message(data2)

            else:
                # cv2.putText(frame, 'no
                data2 = json.dumps({"disease": 'not recognised', "cause": '', "remedie": ''})
                self.ws.write_message(data2)
                # self.ws.write_message('not recognised')

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
        print('opened..')
        global dic_data
        names = None
        data = []
        labels = []

        print(self.request.remote_ip)

        datasets = 'dataset/new'

        print('Training...')

        # Create a list of images and a list of corresponding names
        (images, labels, names, id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    label = id
                    frame = cv2.imread(path, 0)
                    frame = imutils.resize(frame, width=400)
                    images.append(frame)
                    labels.append(int(label))
                id += 1

        # Create a Numpy array from the two lists above
        (images, labels) = [numpy.array(lis) for lis in [images, labels]]

        model = cv2.face.LBPHFaceRecognizer_create()  # cv2.face.LBPHFaceRecognizer_create()
        model.train(images, labels)

        self.cam_thread = CamThread(self, model, names, dic_data)
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
    # get_Host_name_IP()
    dic_data = {}
    disc_pickle_path = 'data.db'
    if os.path.isfile(disc_pickle_path):
        print('')
        disc_pickle_in = open(disc_pickle_path, "rb")
        dic_data = pickle.load(disc_pickle_in)
    else:
        print('data file not found exiting...')
        sys.exit(0)

    # print dic_data

    print([l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1],
                       [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in
                         [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0])
    app.listen(9000)
    autoreload.start()
    for dir, _, files in os.walk('static'):
        [autoreload.watch(dir + '/' + f) for f in files if not f.startswith('.')]
    for dir, _, files in os.walk('templates'):
        [autoreload.watch(dir + '/' + f) for f in files if not f.startswith('.')]
    print('server started at: http://localhost:9000')
    ioloop.IOLoop.instance().start()
