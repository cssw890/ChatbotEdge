# Speech To Text Module
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import wave
import time

INT16_MAX = 32768.0  # Maximum value for 16-bit PCM audio

class SpeechRecognizer:
    def __init__(self):
        self.model = WhisperModel("tiny.en", compute_type="float32")  # Initialize whisper as float32
    
    def recognize_speech(self):
        device_index = 0  # use the device index of the mic you want to use
        samplerate = 16000 
        threshold = 0.02  # Silence threshold
        silence_duration = 0.5  # Duration of silence to consider as end of speech in seconds
        buffer_duration = 0.5  # Duration of buffer in seconds
        max_duration = 8  # Maximum duration to listen in seconds

        print("Adjusting for ambient noise, please wait...")
        with sd.InputStream(device=device_index, channels=1, samplerate=samplerate, dtype='int16') as stream:
            time.sleep(1)  
            print("Listening...")  # Start listening

            audio_data = []
            silence_start = None
            recording_start = time.time()
            
            while True:
                buffer = stream.read(int(buffer_duration * samplerate))[0].flatten()
                audio_data.extend(buffer)
                
                # Check if silence
                if np.abs(buffer).mean() < threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        break
                else:
                    silence_start = None
                
                # Check for maximum duration
                if time.time() - recording_start > max_duration:
                    break
            
            print("Recognizing Stopped")

        # Convert to waveform to process
        audio_data = np.array(audio_data, dtype=np.float32) / INT16_MAX

        # Save the audio to a temporary WAV file
        wav_file = "/tmp/recording.wav"
        with wave.open(wav_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(np.array(audio_data * INT16_MAX, dtype=np.int16))

        segments, info = self.model.transcribe(wav_file)  
        text = " ".join(segment.text for segment in segments)
        print(f"Recognized Speech: {text}")
        return text  # Return Text

if __name__ == "__main__":
    recognizer = SpeechRecognizer()
    recognizer.recognize_speech()