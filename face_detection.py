import cv2
import os

# path from files
cascPath = os.path.dirname(cv2.__file__) + \
    "/data/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

while True:
    # capturando frames da câmera
    ret, frames = video_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Armazenando rostos
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    # Desenhando retangulos ao redor dos rostos reconhecidos
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Mostrando  os frames
    cv2.imshow("Video", frames)
    # Esperando até que a tecla "q" seja pressionado para interromper o programa
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Fechando câmera e as janelas
video_capture.release()
cv2.destroyAllWindows()
