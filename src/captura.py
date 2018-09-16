import cv2 
import numpy as np

#treinamento de detecção de face e olho
classificador = cv2.CascadeClassifier("haarcascades/haarcascade-frontalface-default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascades/haarcascade-eye.xml")

camera = cv2.VideoCapture(0)

amostra = 1
numeroDeAmostras = 25
id = input('Digite seu indentificador: ')
largura, altura = 220, 220
print("Capturanda as faces...")

while (True):
	conectado, imagem = camera.read()
	#Detectar face na imagem
	imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) 

	print( np.average(imagemCinza) )
	#Esacala da imagem e tamanho para detecção de face
	facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(150,150))
	#Marcando o rosto da imagem
	for (x, y, l, a) in facesDetectadas:
		#Verificando a face
		cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
		#Verificando se face tem olhos
		regiao = imagem[y:y + a, x:x + l]
		regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
		olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

		# verificando ser tem olhos 
		for (ox, oy, ol, oa) in olhosDetectados:
			cv2.rectangle(regiao, (ox, oy), (ox, ol, oy, oa), (0, 255, 0), 2)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				if np.average(imagemCinza) > 110:
					imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
					cv2.imwrite("fotos/pessoa."+ str(id) + "." + str(amostra) + ".jpg", imagemFace )
					print("[foto " + str(amostra) + " capturada com sucesso]")
					amostra += 1

	#Mostra a imagem capturada
	cv2.imshow("Face", imagem)
	#Mostra a imagem em tempo 
	cv2.waitKey(1)
	if(amostra >= numeroDeAmostras + 1):
		break


print("Fotos capturadas com sucesso")
#Liberar a memoria 
camera.release()

cv2.destroyAllWindows()