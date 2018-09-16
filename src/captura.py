import cv2 

#treinamento de detecção de face
classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")

camera = cv2.VideoCapture(0)

while (True):
	conectado, imagem = camera.read()

	#Detectar face na imagem
	imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) 
	#sacala da imagem e  tamanho para detecção de face
	facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100,100))

	for (x, y, l, a) in facesDetectadas:
		cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

	#Mostra a imagem capturada
	cv2.imshow("Face", imagem)
	#Mostra a imagem em tempo 
	cv2.waitKey(1)

#Liberar a memoria 
camera.release()

cv2.destroyAllWindows()