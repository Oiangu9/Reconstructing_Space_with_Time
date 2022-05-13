import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# we prepare the audio pulse as numpy array according to the user desires

pulse_duration= 1000
boradcast_sample_rate = 44100
frequency = 18000
envelope_std = pulse_duration/2
gaussian_envelope=True

t = np.linspace(0, pulse_duration,
        pulse_duration*broadcast_sample_rate, False)
note = np.sin( 2*np.pi*frequency*t )
# if a gaussian envelope is desired, it is applied
if gaussian_envelope==True:
    def gaussian(x, mu, stdv):
        return np.exp(-0.5*(x - mu)**2 / stdv**2)/(np.sqrt(2.0*np.pi)*stdv)
    gaussian_f = gaussian(t, mu=pulse_duration/2.0, stdv=envelope_std)
    note = note*gaussian_f

# Ensure that highest value is in 16-bit range
audio = (note / np.max(np.abs(note)))*(2**16-1)
# Convert to 16-bit data
audio = audio.astype(np.int16)

# show the audio pulse
plt.plot(t/broadcast_sample_rate, audio)
plt.show()

recording = sd.playrec(audio, self.recording_sample_rate, channels=1)

#sleep(max(0, self.pulse_duration/2-(time()-begin_t)))

sd.wait()
middle_t=time()
logging.info(f"Captured {frames} images while broadcasting sound")
date = datetime.now()

# save recording
write(f"{self.output_path}/SOUND_ECHO_and_DEPTH/SOUND_RECORDINGS/Echo_{date}.wav", self.recording_sample_rate, recording)  # Save as WAV file


#while((time()-begin_t)<self.pulse_duration): # grab all the possible images
