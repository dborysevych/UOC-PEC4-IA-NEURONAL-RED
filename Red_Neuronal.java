```py
  
import numpy as np



# Definición de los pesos de la red neuronal
weights = np.array([[1, -0.5, 0.5, 1],
                  [-1, 1, 1, -0.5],
                  [1, -0.5, -0.25, 1],
                  [-1, -0.5, 1, -0.25]])
  
# Definición de la función ReLU
def relu(x):
  return np.maximum(0, x)
  
# Definición de la función heaviside
def heaviside(x):
  return np.heaviside(x, 0)
  
# Definición de los datos de entrada
  data = np.array([[4, 8, 2],
                    [7, 2, 7],
                    [0, 9, 7],
                    [1, 4, 1],
                    [1, 1, 1],
                    [2, 3, 2],
                    [6, 0, 1],
                    [6, 5, 2],
                    [0, 9, 3],
                    [8, 3, 1],
                    [0, 2, 3],
                    [4, 2, 2],
                    [3, 8, 3],
                    [0, 4, 8],
                    [9, 10, 3],
                    [7, 5, 2],
                    [1, 10, 2],
                    [9, 10, 9]])
# Definición de las etiquetas verdaderas
true_labels = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1])
  
# Cálculo de la salida de la red neuronal
hidden_layer = relu(np.dot(data, weights[:3]))
  
output_layer = heaviside(np.dot(hidden_layer, weights[-1]))
#print(output_layer)
#print(true_labels)
  
# Cálculo de la matriz de confusión
confusion_matrix = np.zeros((2, 2))
for i in range(len(true_labels)):
  print(true_labels[i] , output_layer[i])
  if true_labels[i] == 1 and output_layer[i] == 1:
    confusion_matrix[0, 0] += 1 # Verdadero positivo
  elif true_labels[i] == 0 and output_layer[i] == 0:
    confusion_matrix[1, 1] += 1 # Verdadero negativo
  elif true_labels[i] == 0 and output_layer[i] == 1:
    confusion_matrix[1, 0] += 1 # Falso positivo
  elif true_labels[i] == 1 and output_layer[i] == 0:
    confusion_matrix[0, 1] += 1 # Falso negativo

  
# Cálculo de la métrica F1
precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
f1 = 2 * (precision * recall) / (precision + recall)

  
print("Matriz de confusión:\n", confusion_matrix)
print("Métrica F1:", f1)

  
```
