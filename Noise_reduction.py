import noisereduce as nr
import numpy as np
import scipy.io.wavfile as wavfile
import pyaudio
import wave

# load data
rate, data = wavfile.read("test.wav")
data = np.asarray(data, dtype=np.float32)
# select section of data that is noise
noisy_part = data[0:1000]
# perform noise reduction
reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=False)
reduced_noise = np.int16(reduced_noise)
reduced_noise = reduced_noise.tostring()

p = pyaudio.PyAudio()
wf = wave.open("output.wav", 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(2)))
wf.setframerate(16000)
wf.writeframes(reduced_noise)
wf.close()