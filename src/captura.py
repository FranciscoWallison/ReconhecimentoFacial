import cv2 

camera = cv2.VideoCapture(0)

while (True):
	conectado, imagem = camera.read()

	#Mostra a imagem capturada
	cv2.imshow("Face", imagem)
	#Mostra a imagem em tempo 
	cv2.waitKey(1)

#Liberar a memoria 
camera.release()

cv2.destroyAllWindows()