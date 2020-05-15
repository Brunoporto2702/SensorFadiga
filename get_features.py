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
from skimage.morphology import reconstruction
from pyexcelerate import Workbook
import traceback

def distancia_dos_labios(formato_rosto):
    labio_superior = formato_rosto[50:53]
    labio_superior = np.concatenate((labio_superior, formato_rosto[61:64]))
    labio_inferior = formato_rosto[56:59]
    labio_inferior = np.concatenate((labio_inferior, formato_rosto[65:68]))
    labio_superior_media = np.mean(labio_superior, axis=0)
    labio_inferior_media = np.mean(labio_inferior, axis=0)
    distancia = abs(labio_superior_media[1] - labio_inferior_media[1])
    return distancia

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
    #detector haar cascade
    retangulos_em_volta_da_face = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE)
    
    #verifica se precisa rotacionar
    if len(retangulos_em_volta_da_face) == 0:
        frame = cv2.putText(frame, "falha", (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame = imutils.rotate(frame, 270)
        retangulos_em_volta_da_face = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE)
   
    #detecta o rosto
    try:
        #selecionar roi 
        (x,y,w,h) = retangulos_em_volta_da_face[0] #posicao rosto detectado
        frame = frame[y:y+h,x:x+w] #corta frame
        frame = imutils.resize(frame, width=300, height=300) #resize frame
        retangulo_face = dlib.rectangle(0, 0, 300, 300) #pega o tamanho 

        formato_rosto = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), retangulo_face)
        formato_rosto = face_utils.shape_to_np(formato_rosto)

        #dectecta infos
        ear, olho_esquerdo, olho_direito = calcula_ear(formato_rosto) #olhos
        distancia_entre_os_labios = distancia_dos_labios(formato_rosto) #boca

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

    return frame, rosto

def desenha_rosto(rosto, frame):
    mascara_olho_esquerdo = cv2.convexHull(rosto['olho_esquerdo']) #desenhar
    mascara_olho_direito = cv2.convexHull(rosto['olho_direito']) #desenhar
    cv2.drawContours(frame, [mascara_olho_esquerdo], -1, (0, 255, 0), 1) #desenhar
    cv2.drawContours(frame, [mascara_olho_direito], -1, (0, 255, 0), 1) #desenhar
    labio = rosto['formato'][48:60] #desenhar
    cv2.drawContours(frame, [labio], -1, (0, 255, 0), 1) #desenhar

def exporta_para_xlsx(df):
    values = [df.columns] + list(df.values)
    wb = Workbook()
    wb.new_sheet('df', data=values)
    wb.save('df.xlsx')


def plot_graph(df):
    for index, coluna in enumerate(df.columns):
        plt.subplot(2,1,index+1)
        plt.plot(range(len(df[coluna].values)), df[coluna].values)
        plt.title(coluna)
    plt.show()
 
def arquivo_features(path_to_video, nome_video):
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    video = FileVideoStream(path_to_video).start()

    resultados = []
    try:
        while True: # for frame in video 
            frame = video.read()
            frame = imutils.resize(frame, width=300, height=300)

            frame, rosto = detecta_rosto(frame, predictor, detector)  #detecta rosto
            if rosto != None:
                # desenha_rosto(rosto, frame)  #desenha rosto
                ear = rosto['ear']
                distancia_entre_os_labios = rosto['distancia_entre_os_labios']
            else:
                ear = -1
                distancia_entre_os_labios = -1

            print('ear: {}'.format(ear))
            print('distancia_entre_os_labios: {}'.format(distancia_entre_os_labios))
            print('\n\n')

            resultado = {
                'ear': ear, 
                'distancia_entre_os_labios': distancia_entre_os_labios
                }
        resultados.append(resultado) 
            
        # cv2.imshow('Camera', frame)
        # cv2.waitKey(1)

    except Exception as e:
        print(e)
       # traceback.print_exc()

    df = pd.DataFrame(resultados)
    
    return df