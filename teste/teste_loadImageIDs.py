import cv2
import os
import numpy as np


eigenface 	= cv2.face.EigenFaceRecognizer_create()
fisherface 	= cv2.face.FisherFaceRecognizer_create()
lbph		= cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
	caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]

	for caminhoImagem in caminhos:
		imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY ) 
		#Divide uma string em strings (split) type Array
		id = int(os.path.split(caminhoImagem) [-1].split('.')[1])
		#Print ID
		print(id)

getImagemComId()