'''
Este algoritmo treina um modelo de rede neural para classificação de imagens de vestuário, como camisas, blusas e sapatos.
'''

# imports da api tf.keras
from __future__ import absolute_import, division, print_function, unicode_literals

# tensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# import do dataset com as imagens das roupas
from keras.datasets import fashion_mnist

# bibliotecas auxiliares
import numpy as np
import matplotlib.pyplot as plt

# funções para plotagem de predições após o treinamento dos dados
def plot_image(i, predictions_array, true_label, img):
  '''mostra a imagem da predição, texto azul para acertos e vermelho para erros'''
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(nomes_classes[predicted_label],
                                100*np.max(predictions_array),
                                nomes_classes[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  '''mostra os valores das predições me barras, azul para acertos e vermelho para erros'''
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


'''
Carregando os dados, o datatset para treinamento possui 60.00 entradas enquanto o dataset para avaliação
da precisão possui 10.000 entradas, cada entrada é uma imagem de baixa resolução de um item de roupas.
os valores dos pixels das imagens variam de 0 a 255 e os labels são um array de inteiros de 0 a 9, onde
cada número representa uma classe de roupa diferente, seguindo o seguinte formato:

0 	Camiseta / Top
1 	Calça
2 	Blusa
3 	Vestido
4 	Jaqueta
5 	Sandalha / Salto
6 	Camisa
7 	Tênis
8 	Bolsa
9 	Bota'''

(imagens_treinamento, labels_treinamento), (imagens_teste, labels_teste) = fashion_mnist.load_data()


nomes_classes = ['Camiseta / Top', 'Calça', 'Blusa', 'Vestido', 'Jaqueta',
               'Sandalha / Salto', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

# verificando as dimensões do dataset
#print(imagens_treinamento.shape)
#print(len(labels_treinamento))
#print(labels_treinamento)
#print(imagens_teste.shape)
#print(len(labels_teste))


# escalonando os valores dos pixels das imagens do dataset para ficarem entre 0 e 1
imagens_treinamento = imagens_treinamento / 255.0
imagens_teste = imagens_teste / 255.0

# visualização das primeiras 25 imagens para verificar que o dataset está corretamente pré-processado antes de iniciar
# o treinamento da rede
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(imagens_treinamento[i], cmap=plt.cm.binary)
    plt.xlabel(nomes_classes[labels_treinamento[i]])
plt.show()

'''
Preparando as camadas da rede, a primeira camada transforma as imagens dimensionais para unidimensional,
As duas outras são camadas de redes neurais com alta densidade de conexão e inteiramente conectadas. 
A primeira delas possui 128 nós (ou neurônios). A segunda (e última) é uma camada de softmax de 10 nós 
que retorna uma matriz de 10 pontuações de probabilidade que somam 1. Cada nó contém uma pontuação que 
indica a probabilidade de a imagem atual pertencer a uma das 10 classes.
''' 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#configuração das métricas deavaliação do treinamento
model.compile(optimizer='adam',                       # atualiza o modelo baseado nos dados visualizados e a função loss
              loss='sparse_categorical_crossentropy', # mensura a precisão
              metrics=['accuracy'])                   # usado para acompanhar o treinamento e os passos de teste, retorna a fração de acertos 

# inicio do treinamento do modelo com 5 epochs
model.fit(imagens_treinamento, labels_treinamento, epochs=5)

# mostra a compração dos resultados
test_loss, test_acc = model.evaluate(imagens_teste,  labels_teste, verbose=2)

print('\nTest accuracy:', test_acc)

# predição de 15 imagens do dataset de testes:
predictions = model.predict(imagens_teste)
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], labels_teste, imagens_teste)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], labels_teste)
plt.tight_layout()
plt.show()
