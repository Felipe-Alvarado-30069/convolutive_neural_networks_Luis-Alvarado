"""
    Main file for the project.  
    
    This file is used to run the project and test the CNN.
"""

import sys
# Importa funciones desde otros módulos del proyecto
from src.managing_images import managing_images
from src.convolution_kernels import convolution_kernel, choose_one_kernel

def run(program_to_run):
    """
        Run the project.
        Ejecuta una de las funciones principales dependiendo del argumento recibido.
    """
    if program_to_run == 'conv_ker':
        # Ejecuta la función que aplica kernels de convolución
        convolution_kernel()
    elif program_to_run == 'mng_imgs':
        # Ejecuta la función para manejo de imágenes
        managing_images()
    elif program_to_run == 'conv_video':
        # Ejecuta la función que permite elegir un kernel para aplicarlo en video
        choose_one_kernel()

# Punto de entrada del programa
if __name__=='__main__':
    # Verifica si se pasó un argumento válido desde la línea de comandos
    if len(sys.argv) > 1 and sys.argv[1] in ['mng_imgs', 'conv_ker', 'conv_video']:
        run(sys.argv[1])  # Llama a la función correspondiente
    else:
        # Muestra mensaje de error si el comando es inválido
        print("Invalid command. Please use one of the following commands:")
        print("mng_imgs, conv_ker, conv_video")
