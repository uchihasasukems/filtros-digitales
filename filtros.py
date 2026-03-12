import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# ===============================
# 1. Definición de la señal
# ===============================

fs = 1000  # Frecuencia de muestreo
t = np.linspace(0, 1, fs)

# Señal compuesta
signal = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)

# Ruido blanco
noise = 0.5 * np.random.randn(len(t))

# Señal con ruido
signal_noise = signal + noise


# ===============================
# 2. Diseño de filtros
# ===============================

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a


def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# ===============================
# 3. Aplicación de filtros
# ===============================

b, a = butter_lowpass(100, fs)
low_filtered = lfilter(b, a, signal_noise)

b, a = butter_highpass(100, fs)
high_filtered = lfilter(b, a, signal_noise)

b, a = butter_bandpass(40, 150, fs)
band_filtered = lfilter(b, a, signal_noise)


# ===============================
# 4. Visualización de resultados
# ===============================

plt.figure(figsize=(12,8))

plt.subplot(4,1,1)
plt.plot(t, signal_noise)
plt.title("Señal original con ruido")

plt.subplot(4,1,2)
plt.plot(t, low_filtered)
plt.title("Filtro Pasa Bajos")

plt.subplot(4,1,3)
plt.plot(t, high_filtered)
plt.title("Filtro Pasa Altos")

plt.subplot(4,1,4)
plt.plot(t, band_filtered)
plt.title("Filtro Pasa Bandas")

plt.tight_layout()
plt.show()