import ast
import pickle
import socket
import sys
import threading
import imutils
import os
import cv2
from tornado import websocket, web, ioloop, autoreload

MAX_FPS = 100


class CamThread(threading.Thread):
    def __init__(self, ws, names, path):
        threading.Thread.__init__(self)
        self.cam = cv2.VideoCapture("http://" + ws.request.remote_ip + ":8080/video")
        self.ws = ws
        self.frame = None
        self.runnable = True
        self.count = names
        self.path = path

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
        print ('img_sig:' + str(message))
        if self.frame is not None:
            cv2.imwrite('%s/%s.jpg' % (self.path, self.count), self.frame)
            self.count += 1
            cv2.putText(self.frame, 'image saved..', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)
            # cv2.putText(frame, 'no
            print ('img saved: ' + str(self.count))
            self.ws.write_message('img saved: ' + str(self.count))

    def stop(self):
        self.runnable = False
        self.cam.release()
        cv2.destroyAllWindows()
        sys.exit(0)


class SocketHandler(websocket.WebSocketHandler):

    def data_received(self, chunk):
        print (chunk)
        # pass

    def __init__(self, *args, **kwargs):
        self.cam_thread = None
        self.cam = None

        """ Initialize the Redis store and framerate monitor. """
        super(SocketHandler, self).__init__(*args, **kwargs)

    def open(self, *args, **kwargs):
        global disease
        print ('opened..')

        print (self.request.remote_ip)

        datasets = 'dataset/new'  # All the faces data will be present this folder
        sub_data = 'train'
        sub_data = disease  # sys.argv[1]

        names = 0

        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
            names = 1
            # main(1)
        else:
            num = []
            for file in os.listdir(path):
                if file.endswith(".jpg"):
                    num.append(int(file.split(".")[0]))

            try:
                print(max(num))
                # main(max(num) + 1)
                names = max(num) + 1
            except (ValueError, TypeError):
                # count = max(num) + 1
                names = 1
                # main(1)

        self.cam_thread = CamThread(self, names, path)
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
        print ('closed..')


app = web.Application([
    (r'/ws', SocketHandler),

])

if __name__ == '__main__':

    data = {}
    disc_pickle_path = 'data.db'
    if os.path.isfile(disc_pickle_path):
        print('')
        disc_pickle_in = open(disc_pickle_path, "rb")
        data = pickle.load(disc_pickle_in)
    else:
        data = {}
        pickle_out = open("data.db", "wb")
        pickle.dump(data, pickle_out, pickle.HIGHEST_PROTOCOL)
        pickle_out.close()

    print (data)

    disease = str(input("Enter Disease: "))

    if disease not in data:
        remedie = str(input("Enter Remedies: "))
        cause = str(input("Enter Cuases: "))
        data[disease] = [remedie, cause]

        pickle_out = open("data.db", "wb")
        pickle.dump(data, pickle_out, pickle.HIGHEST_PROTOCOL)
        pickle_out.close()

    print([l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1],
                       [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in
                         [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0])

    app.listen(9000)
    autoreload.start()
    for dir, _, files in os.walk('static'):
        [autoreload.watch(dir + '/' + f) for f in files if not f.startswith('.')]
    for dir, _, files in os.walk('templates'):
        [autoreload.watch(dir + '/' + f) for f in files if not f.startswith('.')]
    print ('server started at: http://localhost:9000')
    ioloop.IOLoop.instance().start()
