from keras_cv_attention_models import yolor
import cv2 as cv
import time
import os
from pandas import read_table
import numpy as np

files_Path = "C:/Users/choin/OneDrive/바탕 화면/invisible/"
file_name_and_time_lst = []
model = yolor.YOLOR_CSP(pretrained="coco")
df = read_table('coco-labels-paper.txt',index_col=False)
col = str(df.columns[0])

COCO_LABELS = df.to_numpy().reshape(-1,).tolist()
COCO_LABELS.insert(0,col)
webcam = cv.VideoCapture(0)

while True:

    for f_name in os.listdir(f"{files_Path}"):
        written_time = os.path.getctime(f"{files_Path}{f_name}")
        file_name_and_time_lst.append((f_name, written_time))

    sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)

    recent_file = sorted_file_lst[0]
    recent_file_name = recent_file[0]
    time.sleep(0.6)

    img_array = np.fromfile("C:/Users/choin/OneDrive/바탕 화면/invisible/"+recent_file_name, np.uint8)
    frame = cv.imdecode(img_array,cv.IMREAD_COLOR)    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    preds = model(model.preprocess_input(frame))
    bboxs, labels, confidences = model.decode_predictions(preds)[0]
    labels = [label for label in labels.numpy().tolist() if label <= 15]
    label_list = [COCO_LABELS[i] for i in labels]
    label_str = str(label_list)
    print(label_str)
    if 0xFF == ord('q'):
        print('off')
        break
