import logging
import sherpa_onnx
import numpy as np
import queue
import threading
import time
import sounddevice as sd

class TTSModule:
    def __init__(self, model_path, tokens_path, data_dir):
        """
        Initialize the TTSModule with the specified model path, tokens path, and data directory.
        """
        self.tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=model_path,
                    lexicon="",
                    data_dir=data_dir,
                    dict_dir="",
                    tokens=tokens_path,
                ),
                provider="cpu",
                debug=False,
                num_threads=1,
            ),
            rule_fsts="",
            max_num_sentences=1,
        )
        
        # Validate TTS configuration
        if not self.tts_config.validate():
            raise ValueError("Please check your config")

        logging.info("Loading TTS model...")
        # Load the TTS model
        self.tts = sherpa_onnx.OfflineTts(self.tts_config)
        logging.info("Loading TTS model done.")

        self.sample_rate = self.tts.sample_rate  # Set the sample rate for audio playback

    def speak_text(self, text):
        """
        Convert the given text to speech and play it using the sound device.
        """
        buffer = queue.Queue()  # Queue to hold audio samples
        playback_event = threading.Event()  # Event to signal the end of playback
        generation_event = threading.Event()  # Event to signal the end of generation
        playback_started = False
        buffer_lock = threading.Lock()

        def generated_audio_callback(samples: np.ndarray, progress: float):
            """
            Callback function to handle generated audio samples.
            """
            nonlocal playback_started  # Declare playback_started as nonlocal to modify it
            with buffer_lock:
                buffer.put(samples)
            if not playback_started:
                playback_started = True
                logging.info("Start playing ...")
                # Start playback thread
                playback_thread = threading.Thread(target=play_audio)
                playback_thread.start()

            return 1

        def play_audio():
            """
            Play audio from the buffer.
            """
            with sd.OutputStream(channels=1, callback=play_audio_callback, dtype="float32", samplerate=self.sample_rate, blocksize=1024):
                playback_event.wait()  # Wait until playback is complete

        def play_audio_callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
            """
            Callback function for sound device to play audio from the buffer.
            """
            with buffer_lock:
                if buffer.empty() and generation_event.is_set():
                    outdata.fill(0)  # Fill with silence if buffer is empty and generation is complete
                    playback_event.set()  # Signal that playback is complete
                    return

                if not buffer.empty():
                    data = buffer.get()
                    chunk_size = len(data)
                    if chunk_size > frames:
                        buffer.put(data[frames:])  # Put remaining data back in buffer
                        data = data[:frames]
                    outdata[:len(data)] = data.reshape(-1, 1)
                    if len(data) < frames:
                        outdata[len(data):].fill(0)

        logging.info("Start generating ...")
        # Generate audio with the callback to stream in real-time
        self.tts.generate(text, sid=0, speed=1.0, callback=generated_audio_callback)
        logging.info("Finished generating!")
        generation_event.set()  # Signal that generation is complete
        playback_event.wait()  # Wait for the playback to complete

        logging.info("Exiting ...")

# Create an instance of TTSModule with model path info, change accordingly
tts = TTSModule(
    model_path='./vits-piper-en_US-amy-low/en_US-amy-low.onnx', 
    tokens_path='./vits-piper-en_US-amy-low/tokens.txt',
    data_dir='./vits-piper-en_US-amy-low/espeak-ng-data'
)

