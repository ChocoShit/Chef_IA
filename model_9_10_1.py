'''
Modelo 9-10-1 para predicción de rating de recetas
Arquitectura: 9 inputs (fuzzy) -> 10 neuronas (tanh) -> 1 salida (lineal)

ENTRENADO CON:
- Dataset: 15,000 filas muestreadas
- Optimizador: Adam (LR=0.01)
- Epochs: 3000
- Batch: 100
- Normalización: MinMaxScaler
'''

import numpy as np
import pickle
import os

class NeuralNetwork_9_10_1:
    def __init__(self, weights_path='modelo.pkl'):
        """
        Inicializa el modelo 9-10-1
        Carga pesos entrenados desde archivo .pkl (embebido en repo)
        """
        self.inputs_number = 9
        self.input_names = ['cal_baja', 'cal_media', 'cal_alta',
                          'prot_baja', 'prot_media', 'prot_alta',
                          'grasa_baja', 'grasa_media', 'grasa_alta']

        self.output_name = 'rating_estimated'

        # Inicializar pesos por defecto
        np.random.seed(42)
        self.W1 = np.random.randn(9, 10) * 0.1
        self.b1 = np.zeros((1, 10))
        self.W2 = np.random.randn(10, 1) * 0.1
        self.b2 = np.zeros((1, 1))

        # Paramètres de normalisation (valores calculados durante l'entraînement)
        self.y_mean = 3.6299
        self.y_std = 0.5287

        # Charger les poids
        self._load_weights(weights_path)

    def _load_weights(self, path):
        """Charge les poids depuis le fichier .pkl"""
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                self.W1 = data['W1']
                self.b1 = data['b1']
                self.W2 = data['W2']
                self.b2 = data['b2']
                if 'scaler_y' in data:
                    self.scaler_y = data['scaler_y']
                print(f"✅ Pesos cargados desde {path}")
            else:
                print(f"⚠️ Archivo no encontrado: {path}, usando pesos por defecto")
        except Exception as e:
            print(f"⚠️ Error cargando pesos: {e}, usando inicialización por defecto")

    def tanh(self, x):
        """Función de activación tanh"""
        return np.tanh(x)

    def predict(self, inputs):
        """
        Predice el rating_estimated para un lote de inputs

        Parámetros:
        inputs: array-like de forma (n_muestras, 9) o (9,) para una sola muestra
                Valores ya normalizados (fuzzy: baja/media/alta en rango 0-1)

        Retorna:
        rating_estimated (escala 1-5)
        """
        X = np.array(inputs)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != 9:
            raise ValueError(f"Se esperaban 9 inputs, se recibieron {X.shape[1]}")

        # Forward pass
        z1 = X @ self.W1 + self.b1
        a1 = self.tanh(z1)
        z2 = a1 @ self.W2 + self.b2

        # Desescalar a rango original (1-5)
        rating = z2.flatten()

        # Escalar usando los parámetros guardados
        rating_orig = rating * self.y_std + self.y_mean
        rating_orig = np.clip(rating_orig, 1.0, 5.0)

        if X.shape[0] == 1:
            return float(rating_orig[0])
        return [float(r) for r in rating_orig]


def main():
    """Prueba del modelo"""
    model = NeuralNetwork_9_10_1()

    ejemplo = [0.5, 0.3, 0.2, 0.2, 0.5, 0.3, 0.1, 0.4, 0.5]
    rating = model.predict(ejemplo)
    print(f"\nPREDICCIÓN DE TEST:")
    print(f"  Inputs: {ejemplo}")
    print(f"  Rating estimado: {rating:.2f}")


if __name__ == "__main__":
    main()