import languagemodels as lm

class LMChatModule:
    def __init__(self):
        lm.config["max_ram"] = "0.5gb" #Set Max RAM
        self.prompt=""

    def generate_text(self, user_message):
        self.prompt = f"\n\n{user_message}"

        response = lm.do(self.prompt)
        

        return response.strip()


lm_chat = LMChatModule()

if __name__ == "__main__":
    user_message = "How are you?"
    generated_text = lm_chat.generate_text(user_message)
    print("Generated text:", generated_text)