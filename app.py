import gradio as gr
from transformers import pipeline

# Load the pipeline (token classification)
token_classifier = pipeline("token-classification", model="WesScivetti/SNACS_English", aggregation_strategy="simple")

def classify_tokens(text):
    results = token_classifier(text)
    output = ""
    for entity in results:
        output += f"{entity['word']} ({entity['entity_group']}, score={entity['score']:.2f})\n"
    return output.strip()

# Gradio Interface
iface = gr.Interface(
    fn=classify_tokens,
    inputs=gr.Textbox(lines=4, placeholder="Enter a sentence..."),
    outputs="text",
    title="Token Classification with Transformers",
    description="Named Entity Recognition (NER) using Hugging Face Transformers"
)

iface.launch()
