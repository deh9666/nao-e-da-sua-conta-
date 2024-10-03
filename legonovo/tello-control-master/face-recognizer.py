# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--encodings", required=True,
# 	help="path to serialized db of facial encodings")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# ap.add_argument("-d", "--detection-method", type=str, default="cnn",
# 	help="face detection model to use: either `hog` or `cnn`")
# args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings/data.db", "rb").read())

# # Configuração da câmera RealSense
# pipeline = rs.pipeline()  # Inicializa o pipeline RealSense para captura de quadros
# config = rs.config()  # Cria um objeto de configuração para a câmera RealSense
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Habilita o fluxo de cores com determinadas configurações
# profile = pipeline.start(config)  # Inicia o pipeline RealSense com as configurações especificadas

# Tamanho da imagem para o reconhecimento
height, width = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # Define a fonte para o texto
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))

print("[INFO] recognizing faces...")

try:
    while True:
        # # Capturando os quadros da câmera RealSense
        # frames = pipeline.wait_for_frames()  # Espera pela chegada de novos quadros
        # color_frame = frames.get_color_frame()  # Obtém o quadro de cor atual
        # if not color_frame:
        #     continue  # Se não houver quadro de cor, passa para o próximo ciclo do loop

        # imagem = np.asanyarray(color_frame.get_data())  # Converte o quadro de cor para uma matriz NumPy
        conectado, image = camera.read()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte a imagem colorida para escala de cinza

        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)
        
        # initialize the list of names for each face detected
        names = []


        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)
            
            # update the list of names
            names.append(name)
        
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        # show the output image
        cv2.imshow("Image", image)
        # cv2.waitKey(0)
        if cv2.waitKey(1) == ord('q'): break


        # # Deteccao da face baseado no haarcascade
        # faceDetect = face_recognition.compare_faces(data["encodings"], encoding)

        # for (x, y, h, w) in faceDetect:
        #     # Desenhando retangulo da face
        #     cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #     # Detector Olho with face
        #     region = imagem[y:y+h, x:x+w]  # Região de interesse onde o rosto foi detectado
        #     imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)  # Converte a região de interesse para escala de cinza
        #     olhoDetector = detectorOlho.detectMultiScale(imageOlhoGray)  # Detecta olhos na região de interesse

        #     for(ox, oy, oh, ow) in olhoDetector:
        #         # Desenhando retangulo do olho da face detectada
        #         cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)
        #         image = cv2.resize(imageGray[y:y+h, x:x+w], (width, height))  # Redimensiona a imagem para o tamanho especificado

        #         # Fazendo comparacao da imagem detectada
        #         id, confianca = reconhecedor.predict(image)  # Realiza o reconhecimento facial na imagem
        #         if id == 1:
        #             name = 'Deborah'
        #         elif id == 2:
        #             name = 'Thomaz'

        #         else:
        #             name = 'Nao identificado'

        #         # Escrevendo texto no frame
        #         cv2.putText(imagem, name, (x, y + (h + 24)), font, 1, (0, 255, 0))
        #         cv2.putText(imagem, str(confianca), (x, y + (h + 43)), font, 1, (0, 0, 255))

        # # Mostrando frame
        # cv2.imshow("Face", imagem)  # Exibe o quadro com a detecção de rosto e reconhecimento facial
        # if cv2.waitKey(1) == ord('q'): break  # Aguarda a tecla 'q' ser pressionada para encerrar o loop

finally:
    # pipeline.stop()  # Encerra o pipeline RealSense
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas pela OpenCV
