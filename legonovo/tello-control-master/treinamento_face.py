import cv2  # Importa a biblioteca OpenCV
import os  # Importa a biblioteca os para lidar com o sistema operacional
import numpy as np  # Importa a biblioteca NumPy para manipulação de arrays

# Usando 3 algoritmos de reconhecimento facial: Eigenfaces, Fisherfaces e LBPH
eigenface = cv2.face.EigenFaceRecognizer_create()  # Cria um reconhecedor de Eigenfaces
#fisherface = cv2.face.FisherFaceRecognizer_create()  # Cria um reconhecedor de Fisherfaces
#lbph = cv2.face.LBPHFaceRecognizer_create()  # Cria um reconhecedor de LBPH

def getImageWithId():
    '''
        Percorrer diretorio fotos, ler todas imagens com CV2 e organizar
        conjunto de faces com seus respectivos ids
    '''
    pathsImages = [os.path.join('fotos', f) for f in os.listdir('fotos')]  # Lista os caminhos de todas as imagens na pasta 'fotos'
    faces = []  # Inicializa uma lista para armazenar as faces das imagens
    ids = []  # Inicializa uma lista para armazenar os ids correspondentes a cada face

    for pathImage in pathsImages:  # Itera sobre cada caminho de imagem
        imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)  # Lê a imagem e converte para escala de cinza
        id = int(os.path.split(pathImage)[-1].split('.')[2])  # Extrai o id do nome do arquivo

        ids.append(id)  # Adiciona o id à lista de ids
        faces.append(imageFace)  # Adiciona a face à lista de faces

        cv2.imshow("Face", imageFace)  # Mostra a face na tela
        cv2.waitKey(10)  # Espera 10 milissegundos antes de continuar o loop
    return np.array(ids), faces  # Retorna os ids e as faces como arrays NumPy


ids, faces = getImageWithId()  # Chama a função para obter os ids e as faces das imagens

# Gerando classifier do treinamento
print("Treinando....")  # Imprime uma mensagem indicando o início do treinamento
eigenface.train(faces, ids)  # Treina o reconhecedor de Eigenfaces com as faces e os ids correspondentes
eigenface.write('classifier/classificadorEigen.yml')  # Salva o classificador treinado em um arquivo YAML

# fisherface.train(faces, ids)  # Treina o reconhecedor de Fisherfaces com as faces e os ids correspondentes
# fisherface.write('classifier/classificadorFisher.yml')  # Salva o classificador treinado em um arquivo YAML

#lbph.train(faces, ids)  # Treina o reconhecedor de LBPH com as faces e os ids correspondentes
#lbph.write('classifier/classificadorLBPH.yml')  # Salva o classificador treinado em um arquivo YAML
print('Treinamento concluído com sucesso!')  # Imprime uma mensagem indicando que o treinamento foi concluído com sucesso
