import tts_module as tts  # Text to speech (TTS)
import stt_module as stt  # Speech to Text (ASR)
import tg_module as tg  # Generative Text (TG)
import time

class chatBot:
    def __init__(self):
        self.recognizer = stt.SpeechRecognizer()  # Load the speech recognizer
        self.text_generator = tg.lm_chat  # Load the LLM instance
        self.texttospeech = tts.tts  # Load the TTS module instance

    def listen_for_command(self):
        print("Listening for speech...") #Listening in the background
        text_input = self.recognizer.recognize_speech()  # Recognize until the natural end of a sentence

        if text_input: #if text input detected
            response = self.process_command(text_input) #process text
            print(f"Chatbot: {response}")  # Print the response
            self.texttospeech.speak_text(response) #Play through speaker

    def process_command(self, command):
        response = self.text_generator.generate_text(command) #Go through LLM to generate output
        return response

    def start(self):
        print("Starting Chat Robot...") #Starting
        while True:
            self.listen_for_command() 
            time.sleep(1)  # Prevent the loop from running too fast

if __name__ == "__main__":
    chatbot = chatBot()
    try:
        chatbot.start()
    except KeyboardInterrupt:
        print("Chatbot stopped by user.") #Ctrl + C to stopped programme