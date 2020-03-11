# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modify by nkloi@hcmut.edu.vn
# 3/2020

import sys
import time

import cv2

from facenet.face_contrib import *


def add_overlays(frame, faces, frame_rate, colors, confidence=0.4):
    if faces is not None:
        for idx, face in enumerate(faces):
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), colors[idx], 2)
            if face.name and face.prob:
                if face.prob > confidence:
                    class_name = face.name
                else:
                    class_name = 'Unknow'
                    # class_name = face.name
                cv2.putText(frame, class_name, (face_bb[0], face_bb[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            colors[idx], thickness=2, lineType=2)
                cv2.putText(frame, '{:.02f}'.format(face.prob * 100), (face_bb[0], face_bb[3] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], thickness=1, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def run(model_checkpoint, classifier, video_file=None, output_file=None):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    if video_file is not None:
        video_capture = cv2.VideoCapture(video_file)
    else:
        # Use internal camera
        video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    width = frame.shape[1]
    height = frame.shape[0]
    if output_file is not None:
        video_format = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, video_format, 20, (width, height))
    face_recognition = Recognition(model_checkpoint, classifier)
    start_time = time.time()
    colors = np.random.uniform(0, 255, size=(1, 3))
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)
            for i in range(len(colors), len(faces)):
                colors = np.append(colors, np.random.uniform(150, 255, size=(1, 3)), axis=0)
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate, colors)

        frame_count += 1
        cv2.imshow('Video', frame)
        if output_file is not None:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    if output_file is not None:
        out.release()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run('models', 'models/your_model.pkl', video_file='demo.mp4', output_file='demo.avi')
