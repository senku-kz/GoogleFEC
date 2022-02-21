import os

from PIL import Image
import requests
import csv

import sqlite3

import glob
import cv2
import mediapipe as mp
import numpy as np
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
"""
Each line in the CSV files has the following entries:

- URL of image1 (string)
- Top-left column of the face bounding box in image1 normalized by width (float)
- Bottom-right column of the face bounding box in image1 normalized by width (float)
- Top-left row of the face bounding box in image1 normalized by height (float)
- Bottom-right row of the face bounding box in image1 normalized by height (float)

- URL of image2 (string)
- Top-left column of the face bounding box in image2 normalized by width (float)
- Bottom-right column of the face bounding box in image2 normalized by width (float)
- Top-left row of the face bounding box in image2 normalized by height (float)
- Bottom-right row of the face bounding box in image2 normalized by height (float)

- URL of image3 (string)
- Top-left column of the face bounding box in image3 normalized by width (float)
- Bottom-right column of the face bounding box in image3 normalized by width (float)
- Top-left row of the face bounding box in image3 normalized by height (float)
- Bottom-right row of the face bounding box in image3 normalized by height (float)

- Triplet_type (string) - A string indicating the variation of expressions in the triplet.

- Annotator1_id (string) - This is just a string of random numbers that can be used to search for all the samples in the dataset annotated by a particular annotator.
- Annotation1 (integer)
- Annotator2_id (string)
- Annotation2 (integer)
"""

emotions = ('Amusement', 'Anger', 'Awe', 'Boredom', 'Concentration', 'Confusion', 'Contemplation', 'Contempt',
            'Contentment', 'Desire', 'Disappointment', 'Disgust', 'Distress', 'Doubt', 'Ecstasy', 'Elation',
            'Embarrassment', 'Fear', 'Interest', 'Love', 'Neutral', 'Pain', 'Pride', 'Realization', 'Relief',
            'Sadness', 'Shame', 'Surprise', 'Sympathy', 'Triumph')

path_dataset = './dataset'


def get_one_img(file_name='faceexp-comparison-data-test-public.csv'):
    file = open(file_name)
    csvreader = csv.reader(file)
    row = next(csvreader)
    print(row)
    r = requests.get(url=row[0], stream=True)
    print(r.status_code)
    if r.status_code == 200:
        im = Image.open(r.raw)
        im.show()


def read_data_from_file(file_name='faceexp-comparison-data-test-public.csv'):
    file = open(file_name)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        if row[15] == 'ONE_CLASS_TRIPLET':
            rows.append(row)
    file.close()
    print('The number of ONE_CLASS_TRIPLET is {} images.'.format(len(rows)))
    return rows


def get_img_by_url(dataset):
    dataset_url_list = os.listdir(path_dataset)
    for row in dataset:
        for link in (row[0], row[5], row[10]):
            if link.split("/")[-1] not in dataset_url_list:
                # response = requests.get(url=link, stream=True, verify=False)
                try:
                    response = requests.get(url=link, stream=True)
                    print(response.status_code, link)
                    if response.status_code == 200:
                        img = Image.open(response.raw)
                        # img.show()
                        filename = path_dataset + '/' + link.split("/")[-1]
                        img.save(filename)
                        dataset_url_list.append(link.split("/")[-1])
                except:
                    print('Error', link)
    with open('url_list.txt', 'w') as f_url_list:
        f_url_list.write(';'.join(dataset_url_list))


def write_filename_to_file():
    dataset_url_list = os.listdir(path_dataset)
    with open('url_list.txt', 'w') as f_url_list:
        f_url_list.write(';'.join(dataset_url_list))


def sqlite_create_table():
    try:
        sqlite_connection = sqlite3.connect('db.sqlite')
        sqlite_create_table_query = '''CREATE TABLE if not exists m_fec (
                                    id INTEGER PRIMARY KEY,
                                    name text NOT NULL,
                                    path text NOT NULL UNIQUE,
                                    x1 float,
                                    y1 float,
                                    x2 float,
                                    y2 float,
                                    camera text
                                    );'''

        cursor = sqlite_connection.cursor()
        print("База данных подключена к SQLite")
        cursor.execute(sqlite_create_table_query)
        sqlite_connection.commit()
        print("Таблица SQLite создана")

        cursor.close()

    except sqlite3.Error as error:
        print("Ошибка при подключении к sqlite", error)
    finally:
        if (sqlite_connection):
            sqlite_connection.close()
            print("Соединение с SQLite закрыто")
    pass


def sqlite_insert(dataset):
    r = []
    dataset_url_list = os.listdir(path_dataset)
    for row in dataset:
        for link in (row[0:5], row[5:10], row[10:15]):
            if link[0].split("/")[-1] in dataset_url_list:
                filename = link[0].split("/")[-1]
                r.append((filename, link[0], float(link[1]), float(link[2]), float(link[3]), float(link[4]), ))

    try:
        sqlite_connection = sqlite3.connect('db.sqlite')
        sql_insert = '''
        insert or replace into m_fec(name, path, x1, y1, x2, y2) 
        values (?, ?, ?, ?, ?, ?)'''
        cursor = sqlite_connection.cursor()
        print("Conncted to SQLite")
        cursor.executemany(sql_insert, r)
        sqlite_connection.commit()
        print("Rows was inserted into table SQLite ")
        cursor.close()

    except sqlite3.Error as error:
        print("Ошибка при подключении к sqlite", error)
    finally:
        if (sqlite_connection):
            sqlite_connection.close()
            print("Connection to SQLite was closed")
    pass


def sqlite_select():
    sql = 'select * from m_fec where camera is null;'
    upd = 'update m_fec set camera = ? where id = ?;'

    sqlite_connection = sqlite3.connect('db.sqlite')
    cursor = sqlite_connection.cursor()
    cursor_upd = sqlite_connection.cursor()
    for row in cursor.execute(sql):
        camera = head_pose(path_dataset + '/' + row[1])
        cursor_upd.execute(upd, (camera, row[0],))
        sqlite_connection.commit()


def head_pose(path):
    # print('filename:', path)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # cap = cv2.VideoCapture(0)
    # cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    image = cv2.imread(path)

    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False

    # Get the result
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360

            # print('x:y', x, y)

            # See where the user's head tilting
            if y < -10:
                text = "Camera Right (Looking Left)"
                # print("Camera Right (Looking Left)")
                # print("Camera Left (Looking Right)")
            elif y > 10:
                text = "Camera Left (Looking Right)"
                # print("Camera Left (Looking Right)")
                # print("Camera Right (Looking Left)")
            elif x < -10:
                text = "Camera up (Looking Down)"
                # print("Camera up (Looking Down)")
            elif x > 10:
                text = "Camera down (Looking up)"
                # print("Camera up (Looking Down)")
            else:
                text = "Forward"
                # print("Forward")
            print('filename: {}; \t head position: {}'.format(path, text))
            return text




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # read_data_from_file()
    # get_one_img()

    # dataset_url = read_data_from_file()
    # get_img_by_url(dataset_url)
    # write_filename_to_file()

    # sqlite_create_table()
    # sqlite_insert(dataset_url)

    sqlite_select()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
