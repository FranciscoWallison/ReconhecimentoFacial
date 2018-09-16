import cv2
import os
import numpy as np


eigenface 	= cv2.face.EigenFaceRecognizer_create()
fisherface 	= cv2.face.FisherFaceRecognizer_create()
lbph		= cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
	caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]

	faces 	= []
	ids 	= []

	for caminhoImagem in caminhos:
		imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY ) 
		#Divide uma string em strings (split) type Array
		id = int(os.path.split(caminhoImagem) [-1].split('.')[1])

		#Verificar o id com a sua respectiva imagem
		ids.append(id) 
		faces.append(imagemFace) 

	return np.array(ids), faces

ids, faces = getImagemComId()
#Carregando a matriz das imagens
print(faces);
#Carregando a id's
print(ids);


getImagemComId()