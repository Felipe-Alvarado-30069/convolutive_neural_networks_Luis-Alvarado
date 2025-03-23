"""
    Este módulo contiene los kernels de convolución utilizados en las capas convolucionales de la CNN.

    Módulos principales utilizados:
    - numpy (pip install numpy)
    - opencv-python (pip install opencv-python)
    - matplotlib (pip install matplotlib)

    Página de ayuda:
    - https://setosa.io/ev/image-kernels/
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Diccionario de kernels de convolución predefinidos
kernels = {
    "identidad": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]),

    "blur": (1 / 9) * np.ones((3, 3)),

    "gaussiano": (1 / 16) * np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]),

    "sobel_x": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]),

    "sobel_y": np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ]),

    "laplaciano": 3 * np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ]),

    "sharpen": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
}

# Lista de nombres de kernels disponibles
kernel_choose_list = [
    "identidad", 
    "blur", 
    "gaussiano", 
    "sobel_x", 
    "sobel_y", 
    "laplaciano", 
    "sharpen"
]

def convolution_kernel():
    """
        Aplica diferentes kernels de convolución a una imagen.
    """

    # Define ruta actual y ruta de la imagen a procesar
    current_path = Path(__file__).parent
    images_folder = current_path.parent / "images"
    image_path = images_folder / 'example_1.jpg'
    img = cv2.imread(image_path)  # Lee la imagen con OpenCV

    # Convierte la imagen de BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img_rgb.shape)

    # Convierte la imagen a escala de grises
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    print(img_gray.shape)

    # Muestra imagen original y en escala de grises
    axs_orgl = plt.subplots(1, 2, figsize=(16, 8))[1]
    axs_orgl[0].imshow(img_rgb)
    axs_orgl[0].set_title("Imagen original")
    axs_orgl[0].axis('off')

    axs_orgl[1].imshow(img_gray, cmap='gray')
    axs_orgl[1].set_title("Imagen escala de grises")
    axs_orgl[1].axis('off')
    plt.show()

    # Diccionario de kernels con sus nombres descriptivos
    kernels = {
        "Identidad (imagen original)": np.array([
                                            [0, 0, 0],
                                            [0, 1, 0],
                                            [0, 0, 0]]),
        "Blur promedio (para suavizar el ruido)": (1/9)*np.ones((3,3)), 
        "Blur gausiano (suavizado natural)": (1/16)*np.array([
                                            [1, 2, 1],
                                            [2, 4, 2],
                                            [1, 2, 1]]),
        "Sobel Horizontal (detecta bordes horizontales)": 3*np.array([
                                            [-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]]),
        "Sobel Vertical (detecta bordes verticales)": np.array([
                                            [-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]),
        "Laplaciano (detecta bordes)": (3)*np.array([
                                            [0, -1, 0],
                                            [-1, 4, -1],
                                            [0, -1, 0]]),
        "Sharpen (realza detalles)": np.array([
                                            [0, -1, 0],
                                            [-1, 5, -1],
                                            [0, -1, 0]])
    }

    # Crea una figura de subplots para mostrar los resultados
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.flatten()

    # Aplica cada kernel a la imagen y muestra el resultado
    for i, (name, kernel) in enumerate(kernels.items()):
        if 'Sobel' in name or 'Laplaciano' in name:
            img_to_use = img_gray  # Usa imagen en gris para detección de bordes
        else:
            img_to_use = img_rgb   # Usa imagen RGB para filtros generales

        # Aplica el filtro de convolución
        convolved = cv2.filter2D(src=img_to_use, ddepth=-1, kernel=kernel)

        # Determina el tipo de mapa de color
        cmap_type = 'gray' if len(convolved.shape) == 2 else None

        # Muestra la imagen filtrada
        axs[i].imshow(convolved, cmap=cmap_type)
        axs[i].set_title(name)
        axs[i].axis('off')

    # Oculta el último subplot si no se usa
    axs[-1].axis('off')
    plt.tight_layout()
    plt.show()

def convolution_video(kernel):
    """
        Aplica un kernel de convolución a la imagen capturada por la cámara en tiempo real.
    """
    cap = cv2.VideoCapture(0)  # Inicia la cámara

    while True:
        ret, frame = cap.read()  # Lee el frame actual
    
        if not ret:
            break  # Si no se pudo capturar, sale del bucle
    
        filtered = cv2.filter2D(frame, -1, kernel)  # Aplica el kernel
        cv2.imshow("Filtered Video", filtered)  # Muestra el resultado

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Finaliza al presionar la tecla 'q'

    cap.release()  # Libera la cámara
    cv2.destroyAllWindows()  # Cierra la ventana de OpenCV

def choose_one_kernel():
    """
        Pide al usuario elegir un kernel y aplica ese filtro al video en tiempo real.
    """
    kernel_choose = input("Choose one kernel to apply: \n"
                            "0. identidad (imagen original)\n"
                            "1. blur (para suavizar el ruido)\n"
                            "2. gaussiano (suavizado natural)\n"
                            "3. sobel_x (detecta bordes horizontales)\n"
                            "4. sobel_y (detecta bordes verticales)\n"
                            "5. laplaciano (detecta bordes)\n"
                            "6. sharpen (realza detalles):\n"
                            "Choose one option: ")

    # Obtiene el kernel seleccionado y aplica el filtro al video
    kernel = kernels.get(kernel_choose_list[int(kernel_choose)])
    convolution_video(kernel)
