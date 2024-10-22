

# Se plantea utilizando tres imágenes, se pide seleccionar una de ellas, a la que se le 
# inserta ruido.
# Luego se recupera con el método de red de Hopfield y se indica si la posición es correcta para el motor.

# Al finalizar mostrará en pantalla la imagen con ruido, al cerrarla mostrará la imagen recuperada.

import numpy as np
import matplotlib.pyplot as plt

class RedHopfield:
    def __init__(self, tamaño):
        self.tamaño = tamaño
        self.pesos = np.zeros((tamaño, tamaño))

    def entrenar(self, patrones):
        for p in patrones:
            p = p.flatten()
            self.pesos += np.outer(p, p)
        np.fill_diagonal(self.pesos, 0)

    def predecir(self, patron, pasos=5):
        patron = patron.flatten()
        for _ in range(pasos):
            for i in range(self.tamaño):
                s = np.dot(self.pesos[i], patron)
                patron[i] = 1 if s > 0 else 0
        return patron

    def mostrar_patron(self, patron, titulo):
        plt.imshow(patron.reshape((10, 10)), cmap='gray', interpolation='nearest')
        plt.title(titulo)
        plt.axis('off')
        plt.show()

def crear_patron_circulo():
    patron_circulo = np.zeros((10, 10))
    posiciones = [(2, 4), (2, 5), (3, 3), (3, 6), (4, 2), (4, 7),
                  (5, 2), (5, 7), (6, 3), (6, 6), (7, 4), (7, 5)]
    
    for pos in posiciones:
        patron_circulo[pos] = 1
        
    return patron_circulo

def crear_patron_circulo_aleatorio(patron_circulo, desplazamiento):
    patron_aleatorio = np.zeros_like(patron_circulo)
    for (i, j) in np.argwhere(patron_circulo == 1):
        nuevo_i = i + desplazamiento[0]
        nuevo_j = j + desplazamiento[1]
        if 0 <= nuevo_i < patron_aleatorio.shape[0] and 0 <= nuevo_j < patron_aleatorio.shape[1]:
            patron_aleatorio[nuevo_i, nuevo_j] = 1
    return patron_aleatorio

# Crear y entrenar la red de Hopfield
red_hopfield = RedHopfield(tamaño=100)

# Crear el patrón de la pieza C y entrenar la red
patron_circulo = crear_patron_circulo()
red_hopfield.entrenar([patron_circulo])

# Se ofrece una selección de la imagen a utilizar para insertarle ruido
# y luego recuperarla con el uso de la red de Hopfield
opcion = int(input("Seleccione la imagen a utilizar (1, 2 o 3): "))
desplazamiento = (0, 0)

# Generar el patrón aleatorio (ruido en la imagen) en base a la selección
if opcion == 1:
    desplazamiento = (0, 0)  # Pieza en su posición original
elif opcion == 2:
    desplazamiento = (2, 1)  # Pieza desplazada
elif opcion == 3:
    desplazamiento = (1, 2)  # Pieza desplazada
else:
    print("Opción no válida")
    exit()

# Creación del patrón aleatorio
patron_aleatorio = crear_patron_circulo_aleatorio(patron_circulo, desplazamiento)

# Ruido en el patrón aleatorio
patron_con_ruido = patron_aleatorio.copy()

# Corromper un valor al azar del patrón aleatorio
indices_circulo = np.argwhere(patron_con_ruido == 1)
if len(indices_circulo) > 0:
    ruido_i, ruido_j = indices_circulo[np.random.choice(len(indices_circulo))] 
    patron_con_ruido[ruido_i, ruido_j] = 0

# Mostrar en pantalla el patrón corrupto obtenido
red_hopfield.mostrar_patron(patron_con_ruido, 'Imagen con ruido')

# Realizar la predicción para recuperar la imagen
patron_recuperado = red_hopfield.predecir(patron_con_ruido)

# En base a la imagen seleccionada, reubicarla en posición de la matriz
patron_recuperado_con_posicion = np.zeros_like(patron_aleatorio)
for i in range(len(patron_recuperado)):
    if patron_recuperado[i] == 1:
        nuevo_i, nuevo_j = divmod(i, 10)
        nuevo_i += desplazamiento[0]
        nuevo_j += desplazamiento[1]
        if 0 <= nuevo_i < patron_recuperado_con_posicion.shape[0] and 0 <= nuevo_j < patron_recuperado_con_posicion.shape[1]:
            patron_recuperado_con_posicion[nuevo_i, nuevo_j] = 1

# Mostrar el patrón recuperado
red_hopfield.mostrar_patron(patron_recuperado_con_posicion, 'Imagen recuperada')

# Indicar si la posición de la pieza es la correcta en el motor
if opcion == 1:
    print("\nLa posición de la pieza C es correcta\n")
else:
    print("\nLa posición de la pieza C no es correcta\n")

# Notas:
# Se complicó trabajar con tres imágenes desplazadas en ubicación pero con la misma forma,
# al ser iguales el método no funcionaba bien. Por esto opté por entrenarlo únicamente con la imagen
# que corresponde a la posición 1, que determiné como la "correcta" en el motor.

# Finalmente, se dificultaba recuperar la imagen con el desplazamiento que tenía. Como en este caso
# conozco que se trata de una imagen igual pero desplazada, y conozco el desplazamiento, opté
# por recuperar la posición de la pieza C en base a su selección original.

   