from transformers import pipeline

class LLM:
    def __init__(self, model_name="google/flan-t5-small"):
        self.pipe = pipeline(
            "text2text-generation",
            model=model_name
        )

    def __call__(self, prompt):
        output = self.pipe(prompt, max_new_tokens=200)
        return output[0]["generated_text"]
