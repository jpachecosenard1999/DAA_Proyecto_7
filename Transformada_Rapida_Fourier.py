import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import math

def siguiente_potencia_2(n):
    """
    Encuentra la siguiente potencia de 2 mayor o igual a n
    """
    return 2**math.ceil(math.log2(n))

def fft(x):
    """
    Implementación recursiva de la Transformada Rápida de Fourier (FFT)
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    
    # Asegurarse de que la longitud sea potencia de 2
    if N & (N-1) != 0:
        raise ValueError("La longitud del array debe ser potencia de 2")
    
    if N <= 1:
        return x
    
    # Dividir la señal en pares e impares
    pares = fft(x[0::2])
    impares = fft(x[1::2])
    
    # Factores de giro
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    
    # Combinar resultados
    return np.concatenate([
        pares + factor[:N//2] * impares,
        pares + factor[N//2:] * impares
    ])

def ifft(X):
    """
    Transformada inversa de Fourier
    """
    x = fft(np.conjugate(X)) / len(X)
    return np.conjugate(x)

def aplicar_filtro_pasabajas(X, freq_corte, freq_muestreo):
    """
    Aplica un filtro pasa bajas al espectro de frecuencias
    """
    freq = np.fft.fftfreq(len(X), 1/freq_muestreo)
    mascara = np.abs(freq) <= freq_corte
    return X * mascara

def preparar_datos_para_fft(datos):
    """
    Prepara los datos para la FFT rellenando con ceros hasta la siguiente potencia de 2
    """
    N = len(datos)
    N_padded = siguiente_potencia_2(N)
    return np.pad(datos, (0, N_padded - N), 'constant')

def procesar_audio(archivo_entrada, archivo_salida, freq_corte=1000):
    """
    Procesa un archivo de audio aplicando FFT, filtro y IFFT
    """
    # Leer archivo de audio
    freq_muestreo, datos = wavfile.read(archivo_entrada)
    
    # Convertir a float para el procesamiento
    if datos.dtype == np.int16:
        datos = datos.astype(np.float32) / 32768.0
    
    # Si el audio es estéreo, procesar solo el primer canal
    if len(datos.shape) > 1:
        datos = datos[:, 0]
    
    # Guardar la longitud original
    longitud_original = len(datos)
    
    # Preparar datos para FFT
    datos_padded = preparar_datos_para_fft(datos)
    
    # Aplicar FFT
    X = fft(datos_padded)
    
    # Calcular el espectro de frecuencias
    freq = np.fft.fftfreq(len(X), 1/freq_muestreo)
    magnitud = np.abs(X)
    
    # Graficar el espectro de frecuencias
    plt.figure(figsize=(12, 6))
    plt.plot(freq[:len(freq)//2], magnitud[:len(freq)//2])
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud')
    plt.title('Espectro de Frecuencias')
    plt.grid(True)
    plt.show()
    
    # Aplicar filtro pasa bajas
    X_filtrada = aplicar_filtro_pasabajas(X, freq_corte, freq_muestreo)
    
    # Aplicar IFFT
    datos_filtrados = np.real(ifft(X_filtrada))
    
    # Recortar al tamaño original
    datos_filtrados = datos_filtrados[:longitud_original]
    
    # Normalizar y convertir a int16 para guardar
    datos_filtrados = np.int16(datos_filtrados * 32768)
    
    # Guardar archivo de salida
    wavfile.write(archivo_salida, freq_muestreo, datos_filtrados)
