import time
import datetime
import cv2
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor

pathSave1 = os.getcwd()
path1 = pathSave1 + '/Imagenes1/'
path2 = pathSave1 + '/Imagenes2/'
path3 = pathSave1 + '/Imagenes3/'

#Metodo para leer los archivos de texto y capturar el tiempo
def leer_ultimas_lineas(nombre,nombre2,nombre3, n):
    with open(nombre) as fname:
        lineas = [lineas.strip('\n') for lineas in fname.readlines()]
        r1=int(lineas[-n:][0])

    with open(nombre2) as fname:
        lineas2 = [lineas2.strip('\n') for lineas2 in fname.readlines()]
        r2 = int(lineas2[-n:][0])

    with open(nombre3) as fname:
        lineas3 = [lineas3.strip('\n') for lineas3 in fname.readlines()]
        r3 = int(lineas3[-n:][0])

    return [r1,r2,r3]

def detectar(video, path, nombre):
    tiempo_final = 0
    nombreVideo = str(nombre)
    file1 = open(nombreVideo+'.txt',"w")
    cantidad_pistolas = 0
    cantidad_rifles = 0
    cantidad_fuego = 0
    start_time = time.time()
    cap = cv2.VideoCapture(video)
    whT = 320
    confThreshold = 0.5
    nmsThreshold = 0.3

    classesFile = 'obj.names'
    classNames = []

    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    modelConfiguration = 'yolov3.cfg'
    modelWeights = 'yolov3.weights'

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    lista = []
    listP = []
    listR = []
    listF = []
    def findObjects(outputs, img):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []
        tiempo = 0
        nombre = ""
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]

                if confidence > confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            time_milli = cap.get(cv2.CAP_PROP_POS_MSEC)
            if classNames[classIds[i]] != "":
                tiempo = round(time_milli / 1000)
                if tiempo not in lista:
                    nombre = classNames[classIds[i]]

                    cv2.putText(img, f'{nombre.upper()} {int(confs[i] * 100)}%',
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)



        return tiempo, nombre

    #Metodo para graficar los resultados obtenidos por cada video
    def graficarDatos(nombre, tiempoFinal, cantidadPistolas, cantidadRifles, cantidadFuego):
        dura=round(tiempoFinal)#Obtiene solo la parte entera
        print("-------------------------------")
        print("Nombre: " + nombre)
        print("Tiempo: " + str(datetime.timedelta(seconds=dura)))
        print("Pistolas: " + str(cantidadPistolas))
        print("Rifles: " + str(cantidadRifles))
        print("Fuego: " + str(cantidadFuego))
        print("-------------------------------")

        #Escritura de la cantidad de elementos encontrados en el video
        file1.write(str(cantidadPistolas)+'\n')
        file1.write(str(cantidadRifles) + '\n')
        file1.write(str(cantidadFuego) + '\n')
        file1.write(str(dura) + '\n') #cantidad de segundos totales del video procesado.
        file1.close()

        # ---------------------------------GRAFICA CANTIDAD DE ELEMENTOS------------------------------
        #Inicio de la declaración de la grafica
        pathSave = os.getcwd()
        fig = plt.figure(u'Gráfica de barras')  # Figure
        ax = fig.add_subplot(111)  # Axes coordenada dentro de la ventana.

        nombres = ['Pistolas', 'Rifles', 'Fuego']
        datos = [cantidadPistolas, cantidadRifles, cantidadFuego]
        colores = ["blue","red","green"]
        xx = range(len(datos))

        ax.bar(xx, datos,color=colores, width=0.5, align='center')
        ax.set_xticks(xx)
        ax.set_xticklabels(nombres)
        ax.grid(axis='y', color='gray', linestyle='dashed')
        plt.title('Analisis-CantidadElementos ' + nombreVideo)
        plt.savefig(pathSave+'/Graficas/'+nombre+'CantidadApariciones.png')
        #Fin declaración grafica

        #---------------------------------GRAFICA GENERAL DE TIEMPO-----------------------------------

        tiempoGeneral = leer_ultimas_lineas("Video1.txt", "Video2.txt", "Video3.txt", 1)
        # Inicio de la declaración de la GENERAL
        pathSave = os.getcwd()
        plt.clf()
        fig1 = plt.figure(u'Gráfica')  # Figure
        ax1 = fig1.add_subplot(111)  # Axes coordenada dentro de la ventana.

        nombres1 = ['Video1', 'Video2', 'Video3']
        datos1 = [tiempoGeneral[0], tiempoGeneral[1], tiempoGeneral[2]]
        colores1 = ["blue", "red", "green"]
        xx1 = range(len(datos1))

        ax1.bar(xx1, datos1, color=colores1, width=0.5, align='center')
        ax1.set_xticks(xx1)
        ax1.set_xticklabels(nombres1)
        ax1.grid(axis='y', color='gray', linestyle='dashed')

        plt.title('Tiempo General ')
        plt.savefig(pathSave + '/Graficas/' + 'General.png')
        # Fin declaración grafica

        #plt.show()

    aux = 1
    contador = 0
    while True:
        success, img = cap.read()
        if aux % 2 != 0:
            if (success == True):
                contador+=1
                blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
                net.setInput(blob)

                layerNames = net.getLayerNames()
                outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                outputs = net.forward(outputNames)

                tiempo, nombre = findObjects(outputs, img)
                if nombre != "":
                    if tiempo not in lista:
                        if nombre == "Rifle":
                            cantidad_rifles = cantidad_rifles + 1
                            listR.append(tiempo)
                        elif nombre == "Gun":
                            cantidad_pistolas = cantidad_pistolas + 1
                            listP.append(tiempo)
                        elif nombre == "Fire":
                            cantidad_fuego = cantidad_fuego + 1
                            listF.append(tiempo)
                        lista.append(tiempo)

                        cv2.imwrite(path + str(nombre) + str(tiempo) + '.png', img)

                cv2.imshow('Image', img)
                cv2.waitKey(1)
            else:
                # -----------------------------GRAFICA RANGO(TIEMPO) DE APARICION-----------------------------------

                # Graficación del tiempo(rango) en el que aparecen los objetos
                tiempo_final = (time.time() - start_time)
                fig, ax = plt.subplots()
                temperaturas2 = dict(Pistolas=listP, Rifles=listR, Fuegos=listF)

                ax.plot(temperaturas2['Pistolas'],marker = 'o', label='Pistolas')
                ax.plot(temperaturas2['Rifles'],marker = 'o', label='Rifles')
                ax.plot(temperaturas2['Fuegos'],marker = 'o', label='Fuegos')
                ax.legend(loc='upper right')
                ax.grid(axis='y', color='gray', linestyle='dashed')
                pathSave = os.getcwd()
                plt.title('Tiempos de aparición '+ nombreVideo)
                plt.savefig(pathSave+'/Graficas/' + nombreVideo + 'TiemposAparición.png')
                #plt.show()
                graficarDatos(nombreVideo, tiempo_final, cantidad_pistolas, cantidad_rifles, cantidad_fuego)
                break
            aux+=1
        else:
            aux+=1
    return cantidad_pistolas, cantidad_rifles, cantidad_fuego, tiempo_final


if __name__ == '__main__':
    process1 = []
    process2 = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        executor.submit(detectar, '3.mp4', path1, 'Video1')
        executor.submit(detectar, '5.mp4', path2, 'Video2')
        executor.submit(detectar, '6.mp4', path3, 'Video3')


    p1 = 0
    r1 = 0
    f1 = 0

    p2 = 0
    r2 = 0
    f2 = 0

    p3 = 0
    r3 = 0
    f3 = 0

    v1 = open('Video1.txt','r')
    cont = 1
    for line in v1:
        if cont == 1:
            p1 = int(line)
            cont = cont + 1
        elif cont == 2:
            r1 = int(line)
            cont = cont + 1
        elif cont == 3:
            f1 = int(line)
            cont = 1
    v2 = open('Video2.txt','r')
    for line in v2:
        if cont == 1:
            p2 = int(line)
            cont = cont + 1
        elif cont == 2:
            r2 = int(line)
            cont = cont + 1
        elif cont == 3:
            f2 = int(line)
            cont = 1
    v3 = open('Video3.txt', 'r')
    for line in v3:
        if cont == 1:
            p3 = int(line)
            cont = cont + 1
        elif cont == 2:
            r3 = int(line)
            cont = cont + 1
        elif cont == 3:
            f3 = int(line)
            cont = 1

    pTotal = p1 + p2 + p3
    rTotal = r1 + r2 + p3
    fTotal = f1 + f2 + p3


    pathSave = os.getcwd()
    print(pathSave)
    fig = plt.figure(u'Clases Sumadas')  # Figure
    ax = fig.add_subplot(111)  # Axes coordenada dentro de la ventana.

    nombres = ['Pistolas', 'Rifles', 'Fuego']
    datos = [pTotal, rTotal, fTotal]
    xx = range(len(datos))
    colores = ["blue", "red", "green"]
    ax.bar(xx, datos, color=colores, width=0.5, align='center')
    ax.set_xticks(xx)
    ax.set_xticklabels(nombres)
    ax.grid(axis='y', color='gray', linestyle='dashed')
    plt.title("Datos Generales")
    plt.savefig(pathSave+'\Graficas\ApariciónGeneralObjetos.png')


# #------------------------------------------------------------------------




