import gradio as gr
from transformers import pipeline
import spaces

# Load the pipeline (token classification)
#token_classifier = pipeline("token-classification", model="WesScivetti/SNACS_English", aggregation_strategy="simple")


@spaces.GPU  # <-- required for ZeroGPU
def classify_tokens(text):
    token_classifier = pipeline("token-classification", model="WesScivetti/SNACS_English",
                                aggregation_strategy="simple")

    results = token_classifier(text)
    output = ""
    for entity in results:
        output += f"{entity['word']} ({entity['entity_group']}, score={entity['score']:.2f})\n"
    return output.strip()

iface = gr.Interface(
    fn=classify_tokens,
    inputs=gr.Textbox(lines=4, placeholder="Enter text to be classified..."),
    outputs="text",
    title="SNACS Tagging in English",
    description="SNACS Tagging in English"
)

iface.launch()