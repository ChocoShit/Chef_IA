'''
Modelo 9-10-1 para predicción de rating de recetas
Arquitectura: 9 inputs (fuzzy) -> 10 neuronas (tanh) -> 1 salida (lineal)

Entrenado con:
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
    def __init__(self, weights_path='resultados/modelo.pkl', scalers_path='resultados/scalers.pkl'):
        """
        Inicializa el modelo 9-10-1
        Carga pesos y scalers entrenados previamente
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

        # Scalers para normalización
        self.scaler_X = None
        self.scaler_y = None

        # Cargar pesos y scalers
        self._load_model(weights_path, scalers_path)

    def _load_model(self, weights_path, scalers_path):
        """Carga pesos y scalers desde archivos"""
        # Cargar pesos del modelo entrenado
        pkl_path = weights_path
        if not os.path.exists(pkl_path):
            # Buscar en carpeta resultados
            for alt_path in ['resultados/modelo.pkl', 'modelo.pkl']:
                if os.path.exists(alt_path):
                    pkl_path = alt_path
                    break

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            print(f"✅ Pesos cargados desde {pkl_path}")
        else:
            print(f"⚠️ Archivo de pesos no encontrado: {pkl_path}, usando inicialización por defecto")

        # Cargar scalers
        scaler_path = scalers_path
        if not os.path.exists(scaler_path):
            for alt_path in ['resultados/scalers.pkl', 'scalers.pkl']:
                if os.path.exists(alt_path):
                    scaler_path = alt_path
                    break

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
            self.scaler_X = scaler_data['scaler_X']
            self.scaler_y = scaler_data['scaler_y']
            print(f"✅ Scalers cargados desde {scaler_path}")
        else:
            print(f"⚠️ Archivo de scalers no encontrado: {scaler_path}")

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
        # Convertir a numpy array
        X = np.array(inputs)

        # Si es una sola muestra, reshape
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Validar forma
        if X.shape[1] != 9:
            raise ValueError(f"Se esperaban 9 inputs, se recibieron {X.shape[1]}")

        # Forward pass a través de la red
        z1 = X @ self.W1 + self.b1
        a1 = self.tanh(z1)
        z2 = a1 @ self.W2 + self.b2

        # El output está en escala normalizada (0-1)
        # Necesitamos desescalar al rango original de rating (1-5)
        rating = z2.flatten()

        # Escalar manualmente a rango 1-5 (basado en datos de entrenamiento)
        # rating_original = rating * escala + centro
        # Usando los valores del scaler_y guardado
        if self.scaler_y is not None:
            # Invertir la normalización
            rating_orig = self.scaler_y.inverse_transform(rating.reshape(-1, 1)).flatten()
        else:
            # Aproximación: el rating original estaba entre ~1.7 y ~5.0
            # Con MinMaxScaler, el mínimo era ~1.7 y máximo ~5.0
            rating_orig = rating * 3.3 + 1.7  # Aproximación lineal

        # Asegurar que el rating esté en rango válido 1-5
        rating_orig = np.clip(rating_orig, 1.0, 5.0)

        if X.shape[0] == 1:
            return float(rating_orig[0])
        return [float(r) for r in rating_orig]


def main():
    """Prueba del modelo"""
    # Crear instancia
    model = NeuralNetwork_9_10_1()

    # Ejemplo de predicción (valores fuzzy normalizados)
    # [cal_baja, cal_media, cal_alta, prot_baja, prot_media, prot_alta, grasa_baja, grasa_media, grasa_alta]
    ejemplo = [0.5, 0.3, 0.2, 0.2, 0.5, 0.3, 0.1, 0.4, 0.5]

    rating = model.predict(ejemplo)
    print(f"\nPREDICCIÓN DE TEST:")
    print(f"  Inputs: {ejemplo}")
    print(f"  Rating estimado: {rating:.2f}")


if __name__ == "__main__":
    main()