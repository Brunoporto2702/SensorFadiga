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

def distancia_dos_labios(formato_rosto):
    labio_superior = formato_rosto[50:53]
    labio_superior = np.concatenate((labio_superior, formato_rosto[61:64]))
    labio_inferior = formato_rosto[56:59]
    labio_inferior = np.concatenate((labio_inferior, formato_rosto[65:68]))
    labio_superior_media = np.mean(labio_superior, axis=0)
    labio_inferior_media = np.mean(labio_inferior, axis=0)
    distancia = abs(labio_superior_media[1] - labio_inferior_media[1])
    return distancia

#calcula ear
def calcula_ear_para_cada_olho(olho):
    A = dist.euclidean(olho[1], olho[5])
    B = dist.euclidean(olho[2], olho[4])
    C = dist.euclidean(olho[0], olho[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calcula_ear(formato_rosto):
    (lStart, lEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    olho_esquerdo = formato_rosto[lStart:lEnd]
    olho_direito = formato_rosto[rStart:rEnd]
    ear_olho_esquerdo = calcula_ear_para_cada_olho(olho_esquerdo)
    ear_olho_direito = calcula_ear_para_cada_olho(olho_direito)
    ear = (ear_olho_esquerdo + ear_olho_direito) / 2.0
    return (ear, olho_esquerdo, olho_direito)

def detecta_rosto(frame, predictor, detector):
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retangulos_em_volta_da_face = detector(gray_video, 0)
    try:
        retangulo_face = retangulos_em_volta_da_face[0]
        formato_rosto = predictor(gray_video, retangulo_face)
        formato_rosto = face_utils.shape_to_np(formato_rosto)
        ear, olho_esquerdo, olho_direito = calcula_ear(formato_rosto)
        distancia_entre_os_labios = distancia_dos_labios(formato_rosto)

        rosto = {
            'ear': ear, 
            'olho_direito': olho_direito,
            'olho_esquerdo': olho_esquerdo,
            'distancia_entre_os_labios': distancia_entre_os_labios,
            'formato': formato_rosto
            }
    except Exception as e:
        print(e)
        rosto = None

    return rosto

def desenha_rosto(rosto, frame):
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
    #desenha rosto
    if rosto != None:
        desenha_rosto(rosto, frame)

    
    #calcula features
    resultado.append(calcula_features(frame))
    
    cv2.imshow('Camera', frame)
    cv2.waitKey(1)

    #exporta para o csv
    export_to_csv(resultado)