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

def export_to_csv(resultados):
    # exporta para CSV
    pass

def calcula_ear_porcentagem(ear, max_olhoaberto, min_olhofechado):
    ear_porcentagem = 100 * (ear - min_olhofechado)/(max_olhoaberto-min_olhofechado)
    return ear_porcentagem

def add_new_ear(lista_ear_atual,ear):
    lista_ear_atual.append(round(ear,3))
    lista_ear_atual = lista_ear_atual[1:3]
    return lista_ear_atual

def calcula_media(lista_ear_atual):
    media = np.nanmean(lista_ear_atual)
    return media

def calcula_piscadas_por_min(lista_piscadas, num_frames_por_min):
    if len(lista_piscadas) < num_frames_por_min:
        num_frames_por_min = len(lista_piscadas)
    
    piscadas_no_ultimo_min = lista_piscadas[(len(lista_piscadas)- num_frames_por_min):] 

    taxa_piscadas = (max(piscadas_no_ultimo_min) - min(piscadas_no_ultimo_min))

    return taxa_piscadas

detector = dlib.get_frontal_face_detector()  
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

path_to_video = 'Dataset/02/0.mov'
video = FileVideoStream(path_to_video).start()
fps_video_rate = 25 # constante para o caso de vídeo

# video = WebcamVideoStream(src=0).start()    #CAMERA DO PC
fps = FPS().start()         #começa a contar o tempo de teste

resultados = []

calibrado = False
limite_frames_calibracao = 600

lista_ear_calibracao = []
lista_ear_atual = [0]*3
YAWN_THRESH = 25

piscando = False
piscadas = 0
piscadas_por_min = []
lista_piscadas = []

frames = []

# while True: # for frame in video 
for i in range(5000):
    frame = video.read()
    frame = imutils.resize(frame, width=300, height=300)

    rosto = detecta_rosto(frame, predictor, detector)  #detecta rosto
   
    if rosto != None:
        # desenha_rosto(rosto, frame)  #desenha rosto
        ear = rosto['ear']

    lista_ear_atual = add_new_ear(lista_ear_atual,ear)
    ear_filtrado = calcula_media(lista_ear_atual)
    

    if fps._numFrames < limite_frames_calibracao:
        lista_ear_calibracao.append(ear_filtrado)
        max_olhoaberto = np.median(lista_ear_calibracao)
        min_olhofechado = min(lista_ear_calibracao)
        EYE_AR_THRESH = 0.5*max_olhoaberto
    
    elif fps._numFrames >= limite_frames_calibracao:
        #calcula features
        ear_porcentagem = calcula_ear_porcentagem(ear_filtrado, max_olhoaberto, min_olhofechado)

        if ear_filtrado < EYE_AR_THRESH and not piscando:
            piscadas += 1
            piscando = True
        elif ear_filtrado > EYE_AR_THRESH and piscando:
            piscando = False

        num_frames_por_min = 60*fps_video_rate

        if fps._numFrames % num_frames_por_min == 0:
            piscadas_por_min.append(piscadas)
        
        lista_piscadas.append(piscadas)
        piscadas_por_min_atual = calcula_piscadas_por_min(lista_piscadas, num_frames_por_min)

        print('Piscadas total: {}'.format(piscadas))
        print('lista piscadas por min: {}'.format(piscadas_por_min))
        print('piscadas por min atual: {}'.format(piscadas_por_min_atual))
        print('ear%: {}'.format(ear_porcentagem))
        print('ear: {}'.format(ear_filtrado))
        print('\n\n')


        resultado = {
            'ear_porcentagem': ear_porcentagem, 
            'ear_filtrado': ear_filtrado, 
            'ear': ear,
            'EYE_AR_THRESH': EYE_AR_THRESH,
            'piscadas':piscadas,
            'piscadas_por_min_atual': piscadas_por_min_atual,
            }
        resultados.append(resultado) 
        


        

        
    # cv2.imshow('Camera', frame)
    # cv2.waitKey(1)
    fps.update()
    frames.append(fps._numFrames)
            

df = pd.DataFrame(resultados)
for index, coluna in enumerate(df.columns):
    plt.subplot(2,3,index+1)
    plt.plot(range(len(df[coluna].values)), df[coluna].values)
    plt.title(coluna)

plt.show()
    
#exporta para o csv    
export_to_csv(resultados)