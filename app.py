import html
import json
import gradio as gr
import spaces
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import torch
import numpy as np

#Description text for the Gradio interface
DESCR_TOP = """
<h1 style="text-align: center">SNACS Tagging</h1>
"""

DESCR_PART_3 = """
<p style="font-size: 120%;">Enter some text in the box (or use the examples below) and click the Tag! button.
</p>
"""

DESCR_PARA_1 = """<p>🌐 Enter text <b>in any language</b> to analyze the in-context meanings of adpositions/possessives/case markers.
An <b>adposition</b> is a <i>pre</i>position (that precedes a noun, as in English) or a <i>post</i>position (that follows a noun, as in Japanese).
The tagger adds semantic labels from the SNACS tagset to indicate spatial, temporal, and other kinds of relationships. 
See the <a href="https://www.xposition.org/">Xposition site</a> and <a href="https://arxiv.org/abs/1704.02134">PDF manual</a> for details.</p>

<details><summary>Linguistic notes</summary>
<ul>
    <li>Some of the tagged items are single words (like <b><i>to</i></b>); others are multiword expressions (like <b><i>according to</i></b>).</li>
    <li>Possessive markers and possessive pronouns are tagged.</li>
    <li>The English infinitive marker <b><i>to</i></b> is tagged if it marks a purpose.</li>
    <li>Phrasal verb particles (like <b><i>up</i></b> in <b><i>give up</i></b>) are not tagged if the meaning is idiomatic.
    However, words like <b><i>up</i></b>, <b><i>away</i></b>, and <b><i>together</i></b> are tagged if the meaning is spatial
    (“The bird flew <b><i>away</i></b>”).</li>
</ul>
</details>
"""

DESCR_PARA_2 = """<p>🤖 The tagger is a machine learning <a href="https://github.com/WesScivetti/snacs/tree/main">system</a> (specifically XLM-RoBERTa-large)
that has been fine-tuned on manually tagged data in 5 target languages: English, Mandarin Chinese, Hindi, Gujarati, and Japanese.
The system output is not always correct (even if the model’s confidence estimate is close to 100%),
and will likely be less accurate beyond the target languages.</p>

<p><i>This demo by Wesley Scivetti and Nathan Schneider, <a href="http://nert.georgetown.edu">NERT lab</a>, 2025 (<a href="https://github.com/WesScivetti/SNACS_English_Demo">code</a>).</i></p>
"""

# short labels shown on the buttons, long text inserted into the textbox
EXAMPLES = [
    ['When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair.'],   # inserts "example 1" text
    ['Humpty Dumpty was sitting, with his legs crossed like a Turk, on the top of a high wall — such a narrow one that Alice quite wondered how he could keep his balance — and, as his eyes were steadily fixed in the opposite direction, and he didn\'t take the least notice of her, she thought he must be a stuffed figure, after all.'],
    ['In West Philadelphia born and raised\nOn the playground is where I spent most of my days\nChillin\' out, maxin\', relaxin\' all cool\nAnd all shootin\' some b-ball outside of the school\nWhen a couple of guys who were up to no good\nStarted makin\' trouble in my neighborhood\nI got in one little fight and my mom got scared\nAnd said "You\'re movin\' with your auntie and uncle in Bel-Air"'],
    ['En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas, con sus pantuflos de lo mesmo, y los días de entresemana se honraba con su vellorí de lo más fino.']
]
EXAMPLE_LABELS = ['Harry Potter (en)', 'Through the Looking Glass (en)', 'Fresh Prince of Bel-Air (en)', 'Don Quixote (es)']

class FloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class MyPipeline(TokenClassificationPipeline):
    """Custom Pipeline class with custom postprocess function, designed to output proability distribution in addition to top scores
    Inherits from HF TokenClassificationPipeline"""
    def postprocess(self, all_outputs, aggregation_strategy="none", ignore_labels=None):
        try:
            from transformers.pipelines.token_classification import AggregationStrategy
            if isinstance(aggregation_strategy, AggregationStrategy):
                aggregation_strategy = aggregation_strategy.name.lower()
        except Exception:
            pass
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = aggregation_strategy.lower()

        if ignore_labels is None:
            ignore_labels = ["O"]

        id2label = self.model.config.id2label
        label_ids_sorted = sorted(int(k) for k in id2label.keys()) if isinstance(id2label, dict) else list(range(len(id2label)))
        labels_sorted = [id2label[i] for i in label_ids_sorted]

        def _softmax(logits):
            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted = logits - maxes
            exp = np.exp(shifted)
            return exp / exp.sum(axis=-1, keepdims=True)

        all_pre_entities = []
        all_grouped_entities = []

        word_to_chars_map = all_outputs[0].get("word_to_chars_map")
        sentence = all_outputs[0]["sentence"]

        for model_outputs in all_outputs:
            if self.framework == "pt" and model_outputs["logits"][0].dtype in (torch.bfloat16, torch.float16):
                logits = model_outputs["logits"][0].to(torch.float32).numpy()
            else:
                logits = model_outputs["logits"][0].numpy()

            scores = _softmax(logits)

            input_ids = model_outputs["input_ids"][0]
            offset_mapping = model_outputs["offset_mapping"][0] if model_outputs["offset_mapping"] is not None else None
            special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()
            word_ids = model_outputs.get("word_ids")

            if self.framework == "tf":
                input_ids = input_ids.numpy()
                offset_mapping = offset_mapping.numpy() if offset_mapping is not None else None

            pre_entities = self.gather_pre_entities(
                sentence,
                input_ids,
                scores,
                offset_mapping,
                special_tokens_mask,
                aggregation_strategy,  # string is fine
                word_ids=word_ids,
                word_to_chars_map=word_to_chars_map,
            )

            grouped_entities = self.aggregate(pre_entities, aggregation_strategy)
            grouped_entities = [
                e for e in grouped_entities
                if e.get("entity", None) not in ignore_labels
                and e.get("entity_group", None) not in ignore_labels
            ]

            all_pre_entities.extend(pre_entities)
            all_grouped_entities.extend(grouped_entities)

        if len(all_outputs) > 1:
            all_grouped_entities = self.aggregate_overlapping_entities(all_grouped_entities)

        def token_pred_label_id(token_scores: np.ndarray) -> int:
            return int(token_scores.argmax())

        def label_from_entity_dict(ent: dict) -> str | None:
            if "entity_group" in ent and ent["entity_group"] is not None:
                return ent["entity_group"]
            if "entity" in ent and ent["entity"] is not None:
                tag = ent["entity"]
                if tag.startswith(("B-","I-")):
                    return tag[2:]
                return tag
            return None

        def spans_overlap(a_start, a_end, b_start, b_end):
            if a_start is None or a_end is None or b_start is None or b_end is None:
                return False
            return max(a_start, b_start) < min(a_end, b_end)

        pre_tokens = []
        for pe in all_pre_entities:
            pre_tokens.append({
                "start": pe.get("start"),
                "end": pe.get("end"),
                "index": pe.get("index"),
                "scores": pe.get("scores"),
                "pred_id": token_pred_label_id(pe.get("scores")),
                "pred_label": id2label[token_pred_label_id(pe.get("scores"))]
            })

        def average_probs(token_list):
            if not token_list:
                return None
            arr = np.stack([t["scores"] for t in token_list], axis=0)
            avg = np.nanmean(arr, axis=0)
            s = float(avg.sum())
            if s > 0:
                avg = avg / s
            return avg

        results_with_probs = []
        for ent in all_grouped_entities:
            ent_start = ent.get("start")
            ent_end = ent.get("end")
            ent_tag = label_from_entity_dict(ent)

            if aggregation_strategy == "none":
                idx = ent.get("index")
                candidate_tokens = [t for t in pre_tokens if t["index"] == idx] if idx is not None else []
            else:
                overlapping = [t for t in pre_tokens if spans_overlap(ent_start, ent_end, t["start"], t["end"])]
                def strip_bi(lbl): return lbl[2:] if lbl.startswith(("B-","I-")) else lbl
                if ent_tag is not None:
                    overlapping = [t for t in overlapping if strip_bi(t["pred_label"]) == ent_tag]
                candidate_tokens = overlapping

            avg = average_probs(candidate_tokens)
            if avg is None:
                probs_vec = np.zeros((len(labels_sorted),), dtype=float)
                ent_label = ent.get("entity") or ent.get("entity_group")
                if ent_label and ent_label.startswith(("B-","I-")):
                    ent_label = ent_label[2:]
                chosen_i = None
                for i, lab in enumerate(labels_sorted):
                    base = lab[2:] if lab.startswith(("B-","I-")) else lab
                    if ent_label == base:
                        chosen_i = i
                        break
                probs_vec[chosen_i if chosen_i is not None else 0] = 1.0
            else:
                probs_vec = avg

            ent["probabilities"] = {labels_sorted[i]: float(probs_vec[i]) for i in range(len(labels_sorted))}
            results_with_probs.append(ent)

        return results_with_probs

@spaces.GPU
def classify_tokens(text: str, use_canned=False):
    """Main function for SNACS text classification that is called in the huggingface space
    Input: string to be tagged
    Output: HTML styled rendering of tagged outputs
    styled_html1: HTML output with entities grouped
    table_html: Labels from output one along with confidence scores
    styled_html2: HTML output of labels for raw tokenized output (no grouping of subwords or entities). Top 5 label scores are displayed."""

    PALETTE = [  # "#1f77b4",
                  "#ff7f0e",
                  "#2ca02c",
                  "#d62728",
                  "#9467bd",
                  # "#8c564b",
                  "#e377c2",
                  # "#7f7f7f",
                  "#cccc00", # tweaked
                  "#17becf",
                  "#aec7e8",
                  "#ffbb78",
                  "#c49c94",
                  "#c5b0d5",
                  "#98df8a",
                  "#ff9896",
                  "#f7b6d2",
                  # "#c7c7c7",
                  "#f7f777", # tweaked
                  "#9edae5"
              ][::-1]  # reverse-sort to put the lighter colors first

    if not use_canned:
        model_name = "WesScivetti/SNACS_Multilingual"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
        # ONE pipeline; override aggregation per-call
        pipe = MyPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0,
            framework="pt"
        )

        # tagged spans
        results_spans = pipe(text, aggregation_strategy="simple")
        results_spans.sort(key=lambda x: x["start"])

        # per-token + probabilities
        results_tokens = pipe(text, aggregation_strategy="none", ignore_labels=[])
        results_tokens.sort(key=lambda x: x["start"])
        print(results_tokens)
    else:   # canned example to test the output display
        text = "fox in socks"
        results_spans = [{"start": 4, "end": 6, "entity_group": "p.Locus-p.Locus",
                          "score": 0.46, "word": "in"}]
        results_tokens = [
            {"start": 0, "end": 3, "entity": "O", "score": 1,
                "probabilities": {"O": 1}},
            {"start": 4, "end": 6, "entity": "B-p.Locus-p.Locus", "score": 0.46,
                "probabilities": {"B-p.Locus-p.Locus": 0.46, "B-p.Circumstance-p.Circumstance": 0.3, "B-p.Circumstance-p.Locus": 0.2}},
            {"start": 7, "end": 12, "entity": "O", "score": 1,
                "probabilities": {"O": 1}}
            ]

    # color helper that tolerates B-/I- prefixes
    def pick_color(label: str, lbl2color: dict) -> str:
        base = label[2:] if label.startswith(("B-", "I-")) else label
        if base in lbl2color:
            color = lbl2color[base]
        elif base == "O":
            color = "#b0adac"
            lbl2color[base] = color
        else:
            color = PALETTE[len(lbl2color) % len(PALETTE)]
            lbl2color[base] = color
        return color

    def display_label(label: str) -> str:
        """Simplified version of the label to display, removing "p." prefix and un-duplicating supersenses"""
        lab = label.replace("p.", "")
        lab1, lab2 = lab.split("-")
        if lab1==lab2:
            lab = lab1
        else:
            lab = lab1 + "~>" + lab2
        return lab

    # ---------- Output 1: SIMPLE (grouped spans) ----------
    output1, last_idx = "", 0
    lbl2color = {}
    for e in results_spans:
        s, t = e["start"], e["end"]
        lab = e["entity_group"]  # grouped results use entity_group
        short_lab = display_label(lab)
        score = e["score"]
        word = html.escape(text[s:t])
        output1 += html.escape(text[last_idx:s])
        color = pick_color(lab, lbl2color)
        tooltip = f"{short_lab} ({score:.2f})"
        word_with_label = f"{word}"
        output1 += (
            f"<span style='background-color:{color};padding:2px;border-radius:4px;' "
            f"title='{tooltip}'>{word_with_label}</span>"
        )
        last_idx = t
    output1 += html.escape(text[last_idx:])


    output2, last_idx2 = "", 0
    for e in results_tokens:
        s, t = e["start"], e["end"]
        lab = e["entity"]  # NONE returns `entity`
        probs = e["probabilities"]
        word = html.escape(text[s:t])
        output2 += html.escape(text[last_idx2:s])
        color = pick_color(lab, lbl2color)

        top5 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top5_lines = [f"{html.escape(k)}: {v:.2%}" for k, v in top5]
        tooltip = "Top-5&#10;" + "&#10;".join(top5_lines)

        word_with_label = f"{word}_{html.escape(lab)}"
        output2 += (
            f"<span style='background-color:{color};padding:2px;border-radius:4px;' "
            f"title='{tooltip}'>{word_with_label}</span>"
        )
        last_idx2 = t
    output2 += html.escape(text[last_idx2:])

    # (table can use results_simple)
    table_html = "<table style='border-collapse:collapse;font-family:sans-serif;'>"
    table_html += "<tr><th style='border:1px solid #ccc;padding:6px;'>Token</th>"
    table_html += "<th style='border:1px solid #ccc;padding:6px;'>SNACS Label</th>"
    table_html += "<th style='border:1px solid #ccc;padding:6px;'>Confidence</th></tr>"
    for e in results_spans:
        token = html.escape(e["word"])
        lab = e["entity_group"]
        short_lab = display_label(lab)
        score = f"{e['score']:.2f}"
        color = pick_color(lab, lbl2color)
        table_html += (
            "<tr>"
            f"<td style='border:1px solid #ccc;padding:6px;background-color:{color};'>{token}</td>"
            f"<td style='border:1px solid #ccc;padding:6px;background-color:{color};'>{short_lab}</td>"
            f"<td style='border:1px solid #ccc;padding:6px;'>{score}</td>"
            "</tr>"
        )
    table_html += "</table>"

    styled_html1 = f"<div style='font-family:sans-serif;line-height:1.6;'>{output1}</div>"
    styled_html2 = f"<div style='font-family:sans-serif;line-height:1.6;'>{output2}</div>"

    simple_output_data = {"text": text, "entities": [{**e} | {"entity_group": display_label(e["entity_group"])} for e in results_spans]}
    return simple_output_data, json.dumps(results_spans, cls=FloatEncoder), json.dumps(results_tokens, cls=FloatEncoder), styled_html1, table_html, styled_html2
    # except Exception as e:
    #     # Force the real error into the Space logs
    #     import traceback, sys
    #     traceback.print_exc(file=sys.stderr)
    #     # Also show something in the UI so you know it’s the worker, not Gradio
    #     return f"<pre>{html.escape(repr(e))}</pre>", "", ""


#final rendering of the Gradio interface
with gr.Blocks(title="SNACS Tagging") as demo:
    with gr.Row():
        description = gr.HTML(DESCR_TOP)
    
    with gr.Row():
        with gr.Column():
            para1 = gr.HTML(DESCR_PARA_1)
        with gr.Column():
            para2 = gr.HTML(DESCR_PARA_2)

    with gr.Row():
        description = gr.HTML(DESCR_PART_3)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=4, placeholder="Enter a sentence...", label="Input Text"),
            tag_btn = gr.Button("Tag!", variant="primary")
            examples = gr.Examples(EXAMPLES, input_text, example_labels=EXAMPLE_LABELS)
        with gr.Column() as output:
            with gr.Tab("Simple Output"):
                simple_output = gr.HighlightedText(label="Tagged Text")
            with gr.Tab("Detailed Output"):
                output1 = gr.HTML(label="SNACS Tagged Sentence")
                output2 = gr.HTML(label="SNACS Table with Colored Labels")
                output3 = gr.HTML(label="SNACS Tagged Sentence with No Label Aggregation")
            with gr.Tab("JSON Spans"):
                json_spans = gr.Code(language="json")
            with gr.Tab("JSON Tokens"):
                json_tokens = gr.Code(language="json")

    examples.outputs = [simple_output,json_spans,json_tokens,output1,output2,output3]
    tag_btn.click(fn=classify_tokens, inputs=input_text, outputs=examples.outputs)


demo.launch()
