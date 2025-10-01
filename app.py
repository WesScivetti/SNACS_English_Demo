import html
import gradio as gr
import spaces
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import torch
import numpy as np

#Description text for the Gradio interface
DESCRIPTION = """
<p>Enter text <b>in any language</b> to analyze the in-context meanings of adpositions/possessives/case markers.
An <b>adposition</b> is a <i>pre</i>position (that precedes a noun, as in English) or a <i>post</i>position (that follows a noun, as in Japanese).
The tagger adds semantic labels from the SNACS tagset to indicate spatial, temporal, and other kinds of relationships. 
See the <a href="https://www.xposition.org/">Xposition site</a> and <a href="https://arxiv.org/abs/1704.02134">PDF manual</a> for details.</p>

<p>The tagger is a machine learning <a href="https://github.com/WesScivetti/snacs/tree/main">system</a> (specifically XLM-RoBERTa-large)
that has been fine-tuned on manually tagged data in 5 target languages: English, Mandarin Chinese, Hindi, Gujarati, and Japanese.
The system output is not always correct (even if the model’s confidence estimate is close to 100%),
and will likely be less accurate beyond the target languages.</p>

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

<p>Try these examples:
    <a href="#" onclick="document.getElementsByTagName('textarea')[0].value='When Mr. and Mrs. Dursley woke up on the dull, gray Tuesday our story starts, there was nothing about the cloudy sky outside to suggest that strange and mysterious things would soon be happening all over the country. Mr. Dursley hummed as he picked out his most boring tie for work, and Mrs. Dursley gossiped away happily as she wrestled a screaming Dudley into his high chair.';">Harry Potter</a>, 
    <a href="#" onclick="document.getElementsByTagName('textarea')[0].value='Humpty Dumpty was sitting, with his legs crossed like a Turk, on the top of a high wall — such a narrow one that Alice quite wondered how he could keep his balance — and, as his eyes were steadily fixed in the opposite direction, and he didn\'t take the least notice of her, she thought he must be a stuffed figure, after all.';">Through the Looking Glass</a>,
    <a href="#" onclick="document.getElementsByTagName('textarea')[0].value='In West Philadelphia born and raised\nOn the playground is where I spent most of my days\nChillin\' out, maxin\', relaxin\' all cool\nAnd all shootin\' some b-ball outside of the school\nWhen a couple of guys who were up to no good\nStarted makin\' trouble in my neighborhood\nI got in one little fight and my mom got scared\nAnd said &quot;You\'re movin\' with your auntie and uncle in Bel-Air&quot;';">Fresh Prince of Bel-Air</a>, 
    <a href="#" onclick="document.getElementsByTagName('textarea')[0].value='En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas, con sus pantuflos de lo mesmo, y los días de entresemana se honraba con su vellorí de lo más fino.';">Don Quixote</a>
</p>
"""

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
def classify_tokens(text: str):
    """Main function for SNACS text classification that is called in the huggingface space
    Input: string to be tagged
    Output: HTML styled rendering of tagged outputs
    styled_html1: HTML output with entities grouped
    table_html: Labels from output one along with confidence scores
    styled_html2: HTML output of labels for raw tokenized output (no grouping of subwords or entities). Top 5 label scores are displayed."""
    # try:
    color_dict = {'None': '#6adf97',
              'O': '#bebdc9',
              'p.Purpose-p.Purpose': '#51d662',
              'p.SocialRel-p.Gestalt': '#db7ef7',
              'B-p.Cost-p.Cost': '#f4b518',
              'p.Topic-p.Topic': '#f58231',
              'p.Originator-p.Gestalt': '#f25ca8',
              'p.Originator-p.Source': '#d1b041',
              'p.Recipient-p.Goal': '#9c89f5',
              'p.Possessor-p.Possessor': '#ed51e5',
              'p.Gestalt-p.Gestalt': '#bfef45',
              'p.Ancillary-p.Ancillary': '#73f29f',
              'p.ComparisonRef-p.Goal': '#a16afc',
              'p.Source-p.Source': '#5cc334',
              'p.Theme-p.Theme': '#5b88c8',
              'p.Locus-p.Locus': '#fc537d',
              'p.Characteristic-p.Characteristic': '#c95193',
              'p.Explanation-p.Explanation': '#f08dc5',
              'p.OrgMember-p.Possessor': '#e3bd42',
              'p.Goal-p.Goal': '#ffd8b1',
              'p.Manner-p.Manner': '#5a8ef2',
              'p.ComparisonRef-p.ComparisonRef': '#4df5a9',
              'p.Cost-p.Locus': '#fe5990',
              'p.Duration-p.Duration': '#e87ba2',
              'p.Identity-p.Identity': '#d571f0',
              'p.OrgMember-p.Gestalt': '#18fdd1',
              'p.Experiencer-p.Goal': '#9185ff',
              'p.QuantityItem-p.Whole': '#83bafc',
              'p.Whole-p.Gestalt': '#a4fc81',
              'p.PartPortion-p.PartPortion': '#edfc14',
              'p.Time-p.Time': '#91a9ff',
              'p.Approximator-p.Approximator': '#7a65f7',
              'p.Direction-p.Direction': '#c5f249',
              'p.Locus-p.Direction': '#3bed63',
              'p.Instrument-p.Path': '#0ccdda',
              'p.QuantityItem-p.Gestalt': '#d88be2',
              'p.Species-p.Species': '#4dfc63',
              'p.Org-p.Ancillary': '#f7c360',
              'p.Agent-p.Gestalt': '#f373bf',
              'p.SocialRel-p.Ancillary': '#4ee1dc',
              'p.Circumstance-p.Locus': '#fc4c52',
              'p.Circumstance-p.Circumstance': '#69caeb',
              'p.Whole-p.Whole': '#19fa30',
              'p.QuantityItem-p.QuantityItem': '#fadb4d',
              'p.Theme-p.Purpose': '#fc83eb',
              'p.Goal-p.Locus': '#ff96c3',
              'p.Extent-p.Extent': '#79baf7',
              'p.Experiencer-p.Gestalt': '#a79dfa',
              'p.Means-p.Means': '#bbedd1',
              'p.Beneficiary-p.Beneficiary': '#fc6f74',
              'p.Org-p.Beneficiary': '#42ffe4',
              'p.Stimulus-p.Topic': '#effa66',
              'p.Recipient-p.Ancillary': '#a5ff4b',
              'p.Beneficiary-p.Possessor': '#ee98fa',
              'p.Agent-p.Ancillary': '#ebb0ff',
              'p.Theme-p.Gestalt': '#ff5e93',
              'p.StartTime-p.StartTime': '#c285ff',
              'p.Cost-p.Extent': '#b5fff5',
              'p.Manner-p.Source': '#fc6dd4',
              'p.Characteristic-p.Source': '#5bf22c',
              'p.Locus-p.Path': '#d3c136',
              'p.Manner-p.ComparisonRef': '#32cbcb',
              'p.Extent-p.Whole': '#f78392',
              'p.Experiencer-p.Beneficiary': '#8fbaff',
              'p.Theme-p.ComparisonRef': '#ff6bb5',
              'p.Stuff-p.Stuff': '#d187ff',
              'p.Theme-p.Goal': '#edcae1',
              'p.Interval-p.Interval': '#8cfa93',
              'p.Time-p.Whole': '#ecf0a1',
              'p.Stimulus-p.Beneficiary': '#fc88e0',
              'p.Characteristic-p.Locus': '#d190fc',
              'p.Characteristic-p.Extent': '#41f27f',
              'p.EndTime-p.EndTime': '#55edb2',
              'p.Experiencer-p.Ancillary': '#bce155',
              'p.Agent-p.Agent': '#ebfaac',
              'p.PartPortion-p.Source': '#9eb3c3',
              'p.Locus-p.Source': '#c0e1fa',
              'p.Duration-p.Extent': '#ff9ce3',
              'p.Characteristic-p.Identity': '#b8d3f5',
              'p.Possession-p.PartPortion': '#82b8fa',
              'p.Possession-p.Theme': '#b9b2ed',
              'p.Whole-p.Locus': '#b1ffb0',
              'p.Direction-p.Goal': '#9d90cd',
              'p.Gestalt-p.Locus': '#c5b6fa',
              'p.Org-p.Gestalt': '#fae391',
              'p.Stimulus-p.Goal': '#ff7c73',
              'p.Theme-p.Instrument': '#d387fa',
              'p.Stimulus-p.Force': '#fdbfff',
              'p.Beneficiary-p.Theme': '#68fdb4',
              'p.Characteristic-p.Goal': '#fabed4',
              'p.Time-p.Goal': '#ffcc9c',
              'p.Explanation-p.Time': '#a5f757',
              'p.Instrument-p.Manner': '#bac2ff',
              'p.Possession-p.Ancillary': '#fcbb81',
              'p.Instrument-p.Instrument': '#8cc8ff',
              'p.Ensemble-p.Ancillary': '#93fb41',
              'p.Recipient-p.Gestalt': '#6ed4ff',
              'p.Agent-p.Source': '#fc9aca',
              'p.Whole-p.Source': '#d9faac',
              'p.Stimulus-p.Explanation': '#68c4fc',
              'p.Stimulus-p.Direction': '#ffa3d6',
              'p.ComparisonRef-p.Purpose': '#9dff9c',
              'p.ComparisonRef-p.Locus': '#f7b5c5',
              'p.Theme-p.Ancillary': '#ffea82',
              'p.Identity-p.ComparisonRef': '#fce783',
              'p.QuantityItem-p.Stuff': '#a1f649',
              'p.Recipient-p.Direction': '#fccfb3',
              'p.Path-p.Locus': '#86fc89',
              'p.Originator-p.Agent': '#fa9db1',
              'p.Beneficiary-p.Gestalt': '#26eaf0',
              'p.Possessor-p.Ancillary': '#ffc8a8',
              'p.Beneficiary-p.Goal': '#bec1fa',
              'p.OrgMember-p.PartPortion': '#fcb762',
              'p.PartPortion-p.ComparisonRef': '#6fd197',
              'p.Frequency-p.Extent': '#e7f794',
              'p.Beneficiary-p.Direction': '#94e0f7',
              'p.Characteristic-p.Stuff': '#5eeaff',
              'p.Manner-p.Extent': '#f7fc88',
              'p.Cost-p.Cost': '#f4b518',
              'p.Theme-p.Whole': '#60acfc',
              'p.Frequency-p.Frequency': '#f092e6',
              'p.Purpose-p.Locus': '#80e1ac',
              'p.Force-p.Gestalt': '#a3caff',
              'p.Characteristic-p.Ancillary': '#fcdc81',
              'p.ComparisonRef-p.Source': '#ecfa93',
              'p.Org-p.Instrument': '#fcbbd7',
              'p.Theme-p.Characteristic': '#72f7b7',
              'p.Characteristic-p.Topic': '#fa7dc0',
              'p.Locus-p.Goal': '#9a9ffc',
              'p.Locus-p.Whole': '#fc7e84',
              'p.Theme-p.Locus': '#faaba5',
              'p.Frequency-p.Manner': '#6b5831',
              'p.Locus-p.Ancillary': '#f7d081',
              'p.Topic-p.Identity': '#8cfae3',
              'p.Org-p.Goal': '#ff6370',
              'p.SetIteration-p.SetIteration': '#11e7a6',
              'p.PartPortion-p.Goal': '#ee8159',
              'p.ComparisonRef-p.Ancillary': '#f7ef48',
              'p.Force-p.Force': '#48f7da',
              'p.Approximator-p.Extent': '#6fe1f7',
              'p.Manner-p.Stuff': '#86a8f7',
              'p.Path-p.Goal': '#b186f7',
              'p.Explanation-p.Source': '#f264fa',
              'p.Topic-p.Goal': '#fa649b',
              'p.Possession-p.Locus': '#fa7a81',
              'p.Circumstance-p.Path': '#f7fc81',
              'p.Gestalt-p.Source': '#ae86fc',
              'p.Agent-p.Locus': '#fcae95',
              'p.Stimulus-p.Source': '#aa9afc',
              'p.Org-p.Whole': '#93a2fa',
              'p.Org-p.Source': '#fa87da',
              'p.Time-p.Extent': '#b1d4fa',
              'p.Possessor-p.Locus': '#ff5eac',
              'p.Force-p.Source': '#f0fc81',
              'p.Gestalt-p.Topic': '#fa8ca3',
              'p.Cost-p.Manner': '#ff9ebe',
              'p.Means-p.Path': '#a6fa7f',
              'p.Originator-p.Instrument': '#44fe8a',
              'p.PartPortion-p.Instrument': '#acfcfa',
              'p.Possession-p.Possession': '#eabdfc',
              'p.Agent-p.Beneficiary': '#a2e9fa',
              'p.Instrument-p.Locus': '#8facff',
              'p.Instrument-p.Theme': '#84ff7a',
              'p.Duration-p.Gestalt': '#78f3ff',
              'p.Path-p.Path': '#c7c7ff',
              'p.Theme-p.Source': '#80bfff',
              'p.Time-p.Gestalt': '#bcfab6',
              'p.Time-p.Direction': '#ffa6d5',
              'p.Goal-p.Whole': '#9cffdb',
              'p.Explanation-p.Manner': '#f79eb9',
              'p.Time-p.Interval': '#74f7d4',
              'p.Org-p.Locus': '#90b7fc',
              'p.Gestalt-p.Purpose': '#9ff474',
              'p.Stimulus-p.Theme': '#12dfa1',
              'p.Locus-p.Gestalt': '#f2e991',
              'p.Extent-p.Identity': '#b3b3ff',
              'p.ComparisonRef-p.Beneficiary': '#f47ef3',
              'p.Experiencer-p.Agent': '#7cf79e',
              'p.Time-p.Duration': '#d9f763',
              'p.SocialRel-p.Source': '#a48efa',
              'p.Whole-p.Circumstance': '#fa737c',
              'p.Purpose-p.Goal': '#f2f199'}

    PALETTE = [  # "#1f77b4",
                  "#ff7f0e",
                  "#2ca02c",
                  "#d62728",
                  "#9467bd",
                  # "#8c564b",
                  "#e377c2",
                  # "#7f7f7f",
                  "#bcbd22",
                  "#17becf",
                  "#aec7e8",
                  "#ffbb78",
                  "#98df8a",
                  "#ff9896",
                  "#c5b0d5",
                  "#c49c94",
                  "#f7b6d2",
                  # "#c7c7c7",
                  "#dbdb8d",
                  "#9edae5"
              ][::-1]  # reverse-sort to put the lighter colors first
    # lbl2color = {}
    # for tok, lbl in predictions:
    #     if is_snacs(lbl):
    #         if lbl in lbl2color:
    #             color = lbl2color[lbl]
    #         else:
    #             color = PALETTE[len(lbl2color) % len(PALETTE)]
    #             lbl2color[lbl] = color

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

    results_simple = pipe(text, aggregation_strategy="simple")  # output #1
    results_none = pipe(text, aggregation_strategy="none", ignore_labels=[])  # output #2 (per-token + probabilities)
    print(results_none)

    # sort
    sorted_results1 = sorted(results_simple, key=lambda x: x["start"])
    sorted_results2 = sorted(results_none, key=lambda x: x["start"])

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
        #return color_dict.get(label, color_dict.get(base, "#D3D3D3"))

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
    for e in sorted_results1:
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
    for e in sorted_results2:
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
    for e in sorted_results1:
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
    return styled_html1, table_html, styled_html2
    # except Exception as e:
    #     # Force the real error into the Space logs
    #     import traceback, sys
    #     traceback.print_exc(file=sys.stderr)
    #     # Also show something in the UI so you know it’s the worker, not Gradio
    #     return f"<pre>{html.escape(repr(e))}</pre>", "", ""

iface = gr.Interface(
    fn=classify_tokens,
    inputs=gr.Textbox(lines=4, placeholder="Enter a sentence...", label="Input Text"),
    outputs=[
        gr.HTML(label="SNACS Tagged Sentence"),
        gr.HTML(label="SNACS Table with Colored Labels"),
        gr.HTML(label="SNACS Tagged Sentence with No Label Aggregation")
    ],
    title="SNACS Tagging",
    description=DESCRIPTION,
    theme="default"
)
iface.launch()
