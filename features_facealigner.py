#https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
#https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/                bocejo e olhos
#https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/     olhos

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

def final_ear(shape):
    (lStart, lEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = imutils.face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

def calcula_ear_porcentagem(ear, max_olhoaberto, min_olhofechado):
    ear_porcentagem = 100 * (ear - min_olhofechado)/(max_olhoaberto-min_olhofechado)
    return ear_porcentagem

def arquivo_sensor_fadiga(video_arquivo):
    # detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
    detector = dlib.get_frontal_face_detector()  
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # fa = FaceAligner(predictor, desiredFaceWidth=256)

    # ipCamera = "192.168.1.101"                      #//For Camera in the same wifi that MyMax
    # rtsp = "rtsp://admin:Jacare123$@" + ipCamera + ":554/cam/realmonitor?channel=1&subtype=0"
    # vs = WebcamVideoStream(src=rtsp).start()  #CAMERA EXTERNA                              
    vs = WebcamVideoStream(src=0).start()    #CAMERA DO PC

    # vs = FileVideoStream(video_arquivo).start()  #ARQUIVO DE VIDEO
    # cap = cv2.VideoCapture(video_arquivo)

    # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))         #limite de analise total
    parte1 = 301
    parte2 = 601
    frames_calibracao = parte2-1        #depois definimos para analisar corretamente

    #JUNÇÃO DOS DADOS
    lista_ear_atual = [0]*3
    Tempo = [] #*
    Ears = [] #*
    Tempo_calibra = []
    Ears_calibra = []
    tempos_piscada = [] #*
    tempos_cochilando = []
    tempos_piscada_calibra = []
    Bocejo_normal = [] #*
    Bocejo_pesado = []
    PERCLOS_lista = [] #*
    calibrado = 0
    piscou = 0
    piscou1 = 0
    piscou2 = 0
    falha = 0

    PERCLOS_time = 0
    flag6 = 0
    #--------------------------------------------------------------------------------------------------------------------
    fps = FPS().start()         #começa a contar o tempo de teste

    # try:
    while True:
        # time_msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        # print(time_msec)
        # print(fps._numFrames)
        # print(num_frames)
        frame = vs.read()
        frame = imutils.resize(frame, width=300, height=300) ## 200 X 200 FPS=60 SEM VIDEO TEMPO REAL
        # rects = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE)
        rects = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
        if rects == ():
            falha+=1
            frame = imutils.rotate(frame, 270)    #arquivo de video
        # rects = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(50, 50),flags=cv2.CASCADE_SCALE_IMAGE)
        rects = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0)
        # print(rects)
        for rect in rects:
            # rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
            # frame = fa.align(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), rect)
            
            shape = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), rect)
            shape = face_utils.shape_to_np(shape)
            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye [1]
            rightEye = eye[2]
            distance = lip_distance(shape)

            #CALIBRANDO
            if fps._numFrames <= frames_calibracao:
                Tempo_calibra.append(str(datetime.now()))
                Ears_calibra.append(round(ear,3))
                YAWN_THRESH = 25

                max_olhoaberto = np.median(Ears_calibra)
                min_olhofechado = min(Ears_calibra)
                EYE_AR_THRESH = 0.5*max_olhoaberto
            
            #PÓS CALIBRADO
            elif fps._numFrames > frames_calibracao:      
                ear_porcentagem = calcula_ear_porcentagem(ear, max_olhoaberto, min_olhofechado)

                calibrado = 1
                Tempo.append(str(datetime.now().time().hour) +':'+ str(datetime.now().time().minute) +':'+ str(datetime.now().time().second) +':'+ str(datetime.now().time().microsecond))
                Ears.append(round(ear,3))


                #Com os dados da calibração
                EYE_AR_THRESH_calibra1=np.nanmean(Ears_calibra[:parte1])-np.nanstd(Ears_calibra[:parte1])*1.9
                EYE_AR_THRESH_calibra2=np.nanmean(Ears_calibra[parte1:parte2])-np.nanstd(Ears_calibra[parte1:parte2])*1.9
                EYE_AR_THRESH = (EYE_AR_THRESH_calibra1 + EYE_AR_THRESH_calibra2)/2 #ajuste do corte 
                YAWN_THRESH = 25                         

                #PERCLOS EAR = PERCLOS
                PERCLOS_EAR = round((((np.nanmean(Ears_calibra)+np.nanstd(Ears_calibra))-round(ear,3))/((np.nanmean(Ears_calibra)+np.nanstd(Ears_calibra)) - np.nanmin(Ears_calibra))),3) #depois de calibrar pode-se analisar
                PERCLOS_lista.append(PERCLOS_EAR)

                cv2.putText(frame, "Piscadas Normais: {}".format(len(tempos_piscada)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, "Piscadas Demoradas: {}".format(len(tempos_cochilando)), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                #BOCA
                # if (distance > YAWN_THRESH):
                #     #entrada do bocejo
                #     if flag6 == 0:
                #         flag6 = 1
                #         start_bocejo = datetime.now().time()      
                #     #durante o bocejo
                #     bocejo_duracao_now = ((((datetime.now().time().hour-start_bocejo.hour) * 60 + (datetime.now().time().minute-start_bocejo.minute)) * 60 + (datetime.now().time().second-start_bocejo.second))*(10^6) + start_bocejo.microsecond)
                #     if bocejo_duracao_now > 2000000 and bocejo_duracao_now < 5000000: # entre 2 e 5 segundos
                #         estado_bocejo_normal = 1
                #         print('Bocejo duração normal: {}'.format(round(bocejo_duracao,3)))
                #     if bocejo_duracao_now > 5000000: # 5 segundos
                #         estado_bocejo_normal = 0
                #         print('Bocejo duração grande: {}'.format(round(bocejo_duracao,3)))
                #saída do bocejo
                # if (distance < YAWN_THRESH) and flag6 == 1:
                #     bocejo_duracao = ((((datetime.now().time().hour-start_bocejo.hour) * 60 + (datetime.now().time().minute-start_bocejo.minute)) * 60 + (datetime.now().time().second-start_bocejo.second))*(10^6) + start_bocejo.microsecond)
                #     flag6 = 0
                #     if estado_bocejo_normal == 1:
                #         Bocejo_normal.append(bocejo_duracao)
                #     if estado_bocejo_normal == 0:
                #         Bocejo_pesado.append(bocejo_duracao)

                # print(f'média EAR (0 - 1): {media}')
                print(f'EAR porcentagem (0 - 1): {ear_porcentagem}')
                print(f'EAR: {ear}')
                # print(f'PERCLOS EAR (0 - 1): {PERCLOS_EAR}')
                # # print(f'PERCLOS time (0 - 1): {PERCLOS_time}')
                # print(f'Número de piscadas: {len(tempos_piscada)}')
                # print(f'Número de bocejos: {len(Bocejo_normal)}')
                # # print(f'Duração da piscada (ms): {tempos_piscada}')

                #Features em csv
                features = {'Video': [video_arquivo], 'Tempo': [Tempo], 'EAR': [ear], 'EAR porcentagem': [ear_porcentagem], 'PERCLOS EAR': [PERCLOS_EAR], 'PERCLOS time': [PERCLOS_time]} #, 'Duracao piscada normais': [tempos_piscada]}
                dfresults = pd.DataFrame(features)
                dfresults.to_csv("features.csv", encoding = "ISO-8859-1", mode = 'a',sep=';', header = False, index=False)


                ###################################################################################################
            
            #PERCLOS time
            # ear = media
            if ear <= 0.8:
                #entrada do blink em 80%
                if piscou1 == 0:
                    piscou1 = 1
                    t1 = datetime.now().time()

                #entrada do blink em 20%
                if piscou1 == 1:
                    if ear <= 0.2:
                        if piscou2 == 0:
                            piscou2 = 1
                            t2 = datetime.now().time()
                    #saída do blink em 20%
                    if ear >= 0.2 and piscou2 == 1:
                        piscou2 = 0
                        t3t2 = ((((datetime.now().time().hour-t2.hour) * 60 + (datetime.now().time().minute-t2.minute)) * 60 + (datetime.now().time().second-t2.second))*(10^6) + t2.microsecond)
                        
            #saída do blink
            if ear >= 0.8 and piscou1 == 1 and piscou2 == 0:
                piscou1 = 0
                t4t1 = ((((datetime.now().time().hour-t1.hour) * 60 + (datetime.now().time().minute-t1.minute)) * 60 + (datetime.now().time().second-t1.second))*(10^6) + t1.microsecond)
                
                PERCLOS_time = (t3t2/t4t1)*100
                print(PERCLOS_time)

            #PISCADAS
            if ear <= EYE_AR_THRESH:
                #entrada no blink
                if piscou == 0:
                    piscou = 1
                    start_piscada = datetime.now().time()      
                #durante contagem do blink
                deltat_piscada_TR = ((((datetime.now().time().hour-start_piscada.hour)*60 + (datetime.now().time().minute-start_piscada.minute)) * 60 + (datetime.now().time().second-start_piscada.second))*(10^6) + start_piscada.microsecond)
                #se calibrado
                if piscou == 1 and calibrado == 1:
                    if deltat_piscada_TR > (np.nanmean(tempos_piscada_calibra) + np.nanstd(tempos_piscada_calibra)*1.4): #300000):
                        print('ta lentoooo alerta')
                        estado_piscou_normal = 0
                    if deltat_piscada_TR > (np.nanmean(tempos_piscada_calibra) - np.nanstd(tempos_piscada_calibra)) and deltat_piscada_TR < (np.nanmean(tempos_piscada_calibra) + np.nanstd(tempos_piscada_calibra)*1.4): #tem que estar dentro da faixa de tempo esperada para uma piscada normal (precisa ter os dados do calibrado)
                        print('ta normal')
            #saída do blink
            if ear > EYE_AR_THRESH and piscou == 1:
                piscou = 0
                tempo_piscadinha = ((((datetime.now().time().hour-start_piscada.hour) * 60 + (datetime.now().time().minute-start_piscada.minute)) * 60 + (datetime.now().time().second-start_piscada.second))*(10^6) + start_piscada.microsecond)
                if calibrado == 0:
                    tempos_piscada_calibra.append(tempo_piscadinha) #assumindo q na calibracao teremos só piscadas normais
                    # print('calibrando')
                if calibrado == 1 and tempo_piscadinha > (np.nanmean(tempos_piscada_calibra) + np.nanstd(tempos_piscada_calibra)):
                    tempos_cochilando.append(tempo_piscadinha) #delta t de piscadas normais em uma lista pós calibragem
                    print('demorou')
                if calibrado == 1 and tempo_piscadinha > (np.nanmean(tempos_piscada_calibra) - np.nanstd(tempos_piscada_calibra)) and tempo_piscadinha < (np.nanmean(tempos_piscada_calibra) + np.nanstd(tempos_piscada_calibra)*1.2): #tem que estar dentro da faixa de tempo esperada para uma piscada normal (precisa ter os dados do calibrado)
                    tempos_piscada.append(tempo_piscadinha) #delta t de piscadas normais em uma lista pós calibragem
                
                    print('normal')

            fps.update()

            #SHOW
            leftEyeHull = cv2.convexHull(leftEye) #desenhar
            rightEyeHull = cv2.convexHull(rightEye) #desenhar
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) #desenhar
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) #desenhar
            lip = shape[48:60] #desenhar
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1) #desenhar
            cv2.imshow('Camera', frame)
            cv2.waitKey(1)

            #Nao apagar!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # print(f'EAR (0 - 1): {ear}')
            # print(f'PERCLOS (0 - 1): {PERCLOS_EAR}')
            # print(f'Número de piscadas: {len(tempos_piscada)}')
            # print(f'Número de bocejos: {len(Bocejo_normal)}')
            # print(f'Duração da piscada (ms): {tempos_piscada}')

            # #Features em csv
            # features = {'Hora': [datetime.now()], 'Video': [video_arquivo], 'EAR': [ear], 'PERCLOS': [PERCLOS_EAR], 'Duração piscada': [tempos_piscada]}
            # dfresults = pd.DataFrame(results)
            # dfresults.to_csv("results.csv", encoding = "ISO-8859-1", mode = 'a',sep=';', header = False, index=False)


    # except:
    #     print('Fim de vídeo. Trocar o nome do arquivo.')

    tempo_olho_fechado = sum(tempos_piscada)+sum(tempos_cochilando)

    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()
    #--------------------------------------------------------------------------------------------------------------------
    
    
    # #Resultado em csv
    # results = {'Hora': [datetime.now()], 'Video': [video_arquivo], 'Classificacao': [list(labels)[sizes.index(max(sizes))]], 'Porcentagem': [(max(sizes)/sum(sizes))*100], 'Falha de Leitura': [falha]}
    # dfresults = pd.DataFrame(results)
    # dfresults.to_csv("results.csv", encoding = "ISO-8859-1", mode = 'a',sep=';', header = False, index=False)
    
    
    

    return (categoria, porcentagem)


arquivo = 'Dataset/Fold1_part1/Fold1_part1/01/5.mov'
arquivo_sensor_fadiga(arquivo)