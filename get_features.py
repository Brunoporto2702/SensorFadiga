from imutils.video import WebcamVideoStream                                
from imutils.video import FPS   
from imutils.video import FileVideoStream                                            
import imutils
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial import distance as dist
import numpy as np
import dlib
import cv2
import pandas as pd
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

#calcula ear
def final_ear(shape):
    (lStart, lEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def detecta_rosto(frame, predictor, detector):
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retangulos_em_volta_da_face = detector(gray_video, 0)
    try:
        retangulo_face = retangulos_em_volta_da_face[0]

        formato_rosto = predictor(gray_video, retangulo_face)
        formato_rosto = face_utils.shape_to_np(formato_rosto)
        olho = final_ear(formato_rosto)
        ear, olho_esquerdo, olho_direito = olho[0], olho[1], olho[2]

        distancia_entre_os_labios = lip_distance(formato_rosto)

        rosto = {
            'ear': ear, 
            'olho_direito': olho_direito,
            'olho_esquerdo': olho_esquerdo,
            'distancia_entre_os_labios': distancia_entre_os_labios,
            'formato': formato_rosto
            }
    except:
        rosto = None

    return rosto


def mostra_rosto(rosto, frame):
    mascara_olho_esquerdo = cv2.convexHull(rosto['olho_esquerdo']) #desenhar
    mascara_olho_direito = cv2.convexHull(rosto['olho_direito']) #desenhar
    cv2.drawContours(frame, [mascara_olho_esquerdo], -1, (0, 255, 0), 1) #desenhar
    cv2.drawContours(frame, [mascara_olho_direito], -1, (0, 255, 0), 1) #desenhar
    labio = rosto['formato'][48:60] #desenhar
    cv2.drawContours(frame, [labio], -1, (0, 255, 0), 1) #desenhar

def calcula_features(frame):
    # retorna dicionario com as fetures por frame
    pass

def export_to_csv(resultado):
    # exporta para CSV
    pass

detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

path_to_video = 'Dataset/02/0.mov'
video = FileVideoStream(path_to_video).start()

# video = WebcamVideoStream(src=0).start()    #CAMERA DO PC

resultado = []

while True: # for frame in video 
    frame = video.read()
    frame = imutils.resize(frame, width=300, height=300)

    #detecta rosto
    rosto = detecta_rosto(frame, predictor, detector)
    if rosto != None:
        mostra_rosto(rosto, frame)

    resultado.append(calcula_features(frame))
    
    cv2.imshow('Camera', frame)
    cv2.waitKey(1)

    export_to_csv(resultado)