import cv2

detectorFace = cv2.CascadeClassifier("haarcascades/haarcascade-frontalface-default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read('classificadores/classificadorEigen.yml')
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
	conectado, imagem = camera.read()
	imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, 
														minSize=(30,30))

	for (x, y, l, a) in facesDetectadas:
		imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
		cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
		id, confianca = reconhecedor.predict(imagemFace)

		
		
		cv2.putText(imagem, str(id), (x, y + (a+30)), font, 2, (0,0,255) )

	cv2.imshow("Face", imagem)

	if cv2.waitKey(1) == ord('q'):
		break


camera.release()
cv2.destroyAllWindows()