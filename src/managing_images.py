""" 
    Una imagen es una matriz de pixeles, donde cada pixel tiene un número
    que representa un color. 

    En el caso de una imagen a color, cada pixel es un vector de 3 números
    que representan los colores RGB (Red, Green, Blue). Cada número va de 0 a 255.

    En el caso de una imagen en escala de grises, cada pixel es un número
    entre 0 y 255, donde 0 es negro y 255 blanco.

    Trabajar con imágenes en escala de grises siempre y cuando el color no importe, 
    ya que serán más fáciles de procesar y más rápidas de entrenar.

    Definir siempre un tamaño de imagen fijo.
    (28, 28, 3) a color
    (28, 28) escala de grises.  
    
    Normalizar las imágenes para que los valores de los pixeles
    estén entre 0 y 1.

    Aumentar el tamaño del dataset con técnicas de data augmentation.
    - Rotar la imagen
    - Hacer zoom
    - Cambiar la iluminación
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def managing_images():
    """
       Esta función nos ayuda a manejar una imagen.
    """

    # Define la ruta actual y la ruta de la imagen
    current_path = Path(__file__).parent
    images_folder = current_path.parent / "images"
    image_path = images_folder / 'example_1.jpg'

    # Carga la imagen con OpenCV
    img = cv2.imread(image_path)
    print(img.shape)  # Muestra la forma (dimensiones) de la imagen original

    # Muestra la imagen en su formato original (BGR)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Convierte la imagen de BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Extrae los canales individuales de color
    red = img_rgb[:,:,0]
    green = img_rgb[:,:,1]
    blue = img_rgb[:,:,2]

    # Crea una matriz de ceros del mismo tamaño de la imagen
    aux_dim = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))

    # Combina cada canal con ceros en los demás para visualizar por separado
    red_color = np.dstack((red, aux_dim, aux_dim)).astype(np.uint8)
    green_color = np.dstack((aux_dim, green, aux_dim)).astype(np.uint8)
    blue_color = np.dstack((aux_dim, aux_dim, blue)).astype(np.uint8)

    # Crea una figura con 3 subplots para mostrar los tres canales
    axs_orgl = plt.subplots(1, 3, figsize=(16, 8))[1]

    axs_orgl[0].imshow(red_color)
    axs_orgl[0].set_title("Imagen Rojo")
    axs_orgl[0].axis('off')

    axs_orgl[1].imshow(green_color)
    axs_orgl[1].set_title("Imagen Verde")
    axs_orgl[1].axis('off')

    axs_orgl[2].imshow(blue_color)
    axs_orgl[2].set_title("Imagen Azul")
    axs_orgl[2].axis('off')

    plt.show()

    # Crea una versión en negativo de la imagen RGB
    img_rgb_neg = 255 - img_rgb
    plt.imshow(img_rgb_neg)
    plt.axis('off')
    plt.show()

    # Convierte la imagen a escala de grises
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    print(img_gray.shape)  # Muestra la forma de la imagen en gris

    # Muestra la imagen en escala de grises
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    plt.show()

    # Crea el negativo de la imagen en escala de grises
    img_gray_neg = 255 - img_gray
    plt.imshow(img_gray_neg, cmap='gray')
    plt.axis('off')
    plt.show()

    # Disminuye la resolución de la imagen (efecto tipo posterizado)
    img_res_32 = (img_rgb//32)*32
    plt.imshow(img_res_32)
    plt.axis('off')
    plt.show()

    # Otra resolución reducida con pasos más grandes
    img_res_128 = (img_rgb//128)*128
    plt.imshow(img_res_128)
    plt.axis('off')
    plt.show()

    # Aplica un recorte para simular zoom (zoom-in)
    plt.imshow(img_rgb[100:200, 100:200])
    plt.axis('off')
    plt.show()
