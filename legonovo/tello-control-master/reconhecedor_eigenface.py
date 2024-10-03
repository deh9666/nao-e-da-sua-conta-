import cv2  # Importa a biblioteca OpenCV para manipulação de vídeo e imagem
import numpy as np  # Importa a biblioteca NumPy para operações matemáticas
# import pyrealsense2 as rs  # Importa a biblioteca RealSense para trabalhar com a câmera RealSense

# Carregando os classificadores Haarcascade para detecção de rosto e olhos
detectorFace = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
detectorOlho = cv2.CascadeClassifier('cascade/haarcascade-eye.xml')

# Carregando o reconhecedor de Eigen Faces para reconhecimento facial
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classifier/classificadorEigen.yml")

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

try:
    while True:
        # # Capturando os quadros da câmera RealSense
        # frames = pipeline.wait_for_frames()  # Espera pela chegada de novos quadros
        # color_frame = frames.get_color_frame()  # Obtém o quadro de cor atual
        # if not color_frame:
        #     continue  # Se não houver quadro de cor, passa para o próximo ciclo do loop

        # imagem = np.asanyarray(color_frame.get_data())  # Converte o quadro de cor para uma matriz NumPy
        conectado, imagem = camera.read()
        imageGray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # Converte a imagem colorida para escala de cinza

        # Deteccao da face baseado no haarcascade
        faceDetect = detectorFace.detectMultiScale(
            imageGray,
            scaleFactor=1.5,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE )

        for (x, y, h, w) in faceDetect:
            # Desenhando retangulo da face
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Detector Olho with face
            region = imagem[y:y+h, x:x+w]  # Região de interesse onde o rosto foi detectado
            imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)  # Converte a região de interesse para escala de cinza
            olhoDetector = detectorOlho.detectMultiScale(imageOlhoGray)  # Detecta olhos na região de interesse

            for(ox, oy, oh, ow) in olhoDetector:
                # Desenhando retangulo do olho da face detectada
                cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)
                image = cv2.resize(imageGray[y:y+h, x:x+w], (width, height))  # Redimensiona a imagem para o tamanho especificado

                # Fazendo comparacao da imagem detectada
                id, confianca = reconhecedor.predict(image)  # Realiza o reconhecimento facial na imagem
                if id == 1:
                    name = 'Deborah'
                elif id == 2:
                    name = 'Thomaz'

                else:
                    name = 'Nao identificado'

                # Escrevendo texto no frame
                cv2.putText(imagem, name, (x, y + (h + 24)), font, 1, (0, 255, 0))
                cv2.putText(imagem, str(confianca), (x, y + (h + 43)), font, 1, (0, 0, 255))

        # Mostrando frame
        cv2.imshow("Face", imagem)  # Exibe o quadro com a detecção de rosto e reconhecimento facial
        if cv2.waitKey(1) == ord('q'): break  # Aguarda a tecla 'q' ser pressionada para encerrar o loop

finally:
    # pipeline.stop()  # Encerra o pipeline RealSense
    cv2.destroyAllWindows()  # Fecha todas as janelas abertas pela OpenCV
