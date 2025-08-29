import html
import gradio as gr
import spaces
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import torch
import numpy as np

# --- Your custom pipeline, no decorator on classes/helpers ---
class MyPipeline(TokenClassificationPipeline):
    def postprocess(self, all_outputs, aggregation_strategy="none", ignore_labels=None):
        # Normalize aggregation_strategy to a lowercase string
        try:
            # Handle HF enum
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

# --- Gradio callback: put ALL model work inside this GPU-decorated function ---

@spaces.GPU
def classify_tokens(text: str):
    try:
        color_dict = {'None': '#6adf97',
                  'O': '#f18621',
                  'p.Purpose-p.Purpose': '#3cb44b',
                  'p.SocialRel-p.Gestalt': '#911eb4',
                  'B-p.Cost-p.Cost': '#f4b518',
                  'p.Topic-p.Topic': '#f58231',
                  'p.Originator-p.Gestalt': '#f25ca8',
                  'p.Originator-p.Source': '#a08323',
                  'p.Recipient-p.Goal': '#725be0',
                  'p.Possessor-p.Possessor': '#f032e6',
                  'p.Gestalt-p.Gestalt': '#bfef45',
                  'p.Ancillary-p.Ancillary': '#73f29f',
                  'p.ComparisonRef-p.Goal': '#a16afc',
                  'p.Source-p.Source': '#5cc334',
                  'p.Theme-p.Theme': '#5b88c8',
                  'p.Locus-p.Locus': '#e6194B',
                  'p.Characteristic-p.Characteristic': '#ba2d7a',
                  'p.Explanation-p.Explanation': '#cc2374',
                  'p.OrgMember-p.Possessor': '#e3bd42',
                  'p.Goal-p.Goal': '#ffd8b1',
                  'p.Manner-p.Manner': '#3a70d6',
                  'p.ComparisonRef-p.ComparisonRef': '#4df5a9',
                  'p.Cost-p.Locus': '#fe5990',
                  'p.Duration-p.Duration': '#e87ba2',
                  'p.Identity-p.Identity': '#cb49ed',
                  'p.OrgMember-p.Gestalt': '#18fdd1',
                  'p.Experiencer-p.Goal': '#400043',
                  'p.QuantityItem-p.Whole': '#5f3ba4',
                  'p.Whole-p.Gestalt': '#497114',
                  'p.PartPortion-p.PartPortion': '#edfc14',
                  'p.Time-p.Time': '#4363d8',
                  'p.Approximator-p.Approximator': '#553ee1',
                  'p.Direction-p.Direction': '#687447',
                  'p.Locus-p.Direction': '#12b336',
                  'p.Instrument-p.Path': '#0ccdda',
                  'p.QuantityItem-p.Gestalt': '#d88be2',
                  'p.Species-p.Species': '#4dfc63',
                  'p.Org-p.Ancillary': '#6a5b9c',
                  'p.Agent-p.Gestalt': '#f373bf',
                  'p.SocialRel-p.Ancillary': '#4ee1dc',
                  'p.Circumstance-p.Locus': '#38abe5',
                  'p.Circumstance-p.Circumstance': '#69caeb',
                  'p.Whole-p.Whole': '#00d816',
                  'p.QuantityItem-p.QuantityItem': '#dbbc2d',
                  'p.Theme-p.Purpose': '#cb56ba',
                  'p.Goal-p.Locus': '#b3597f',
                  'p.Extent-p.Extent': '#5cadfa',
                  'p.Experiencer-p.Gestalt': '#8275f4',
                  'p.Means-p.Means': '#b1bfb7',
                  'p.Beneficiary-p.Beneficiary': '#0e9582',
                  'p.Org-p.Beneficiary': '#c48ea7',
                  'p.Stimulus-p.Topic': '#a6af3a',
                  'p.Recipient-p.Ancillary': '#a5ff4b',
                  'p.Beneficiary-p.Possessor': '#c941dc',
                  'p.Agent-p.Ancillary': '#d18ce9',
                  'p.Theme-p.Gestalt': '#b71c4f',
                  'p.StartTime-p.StartTime': '#9b3cf9',
                  'p.Cost-p.Extent': '#117f70',
                  'p.Manner-p.Source': '#460233',
                  'p.Characteristic-p.Source': '#41c518',
                  'p.Locus-p.Path': '#d3c136',
                  'p.Manner-p.ComparisonRef': '#32cbcb',
                  'p.Extent-p.Whole': '#94454f',
                  'p.Experiencer-p.Beneficiary': '#1f2d98',
                  'p.Theme-p.ComparisonRef': '#ef3f97',
                  'p.Stuff-p.Stuff': '#9919e8',
                  'p.Theme-p.Goal': '#d7c6d1',
                  'p.Interval-p.Interval': '#042206',
                  'p.Time-p.Whole': '#ecf0a1',
                  'p.Stimulus-p.Beneficiary': '#af168a',
                  'p.Characteristic-p.Locus': '#ac54e6',
                  'p.Characteristic-p.Extent': '#0ec04c',
                  'p.EndTime-p.EndTime': '#29e89e',
                  'p.Experiencer-p.Ancillary': '#bce155',
                  'p.Agent-p.Agent': '#aac43b',
                  'p.PartPortion-p.Source': '#9eb3c3',
                  'p.Locus-p.Source': '#7121d7',
                  'p.Duration-p.Extent': '#ca1096',
                  'p.Characteristic-p.Identity': '#345c8d',
                  'p.Possession-p.PartPortion': '#e592aa',
                  'p.Possession-p.Theme': '#a59bec',
                  'p.Whole-p.Locus': '#0bc209',
                  'p.Direction-p.Goal': '#9d90cd',
                  'p.Gestalt-p.Locus': '#97f830',
                  'p.Org-p.Gestalt': '#2f2c3c',
                  'p.Stimulus-p.Goal': '#c40f02',
                  'p.Theme-p.Instrument': '#a312ed',
                  'p.Stimulus-p.Force': '#d98ddb',
                  'p.Beneficiary-p.Theme': '#68fdb4',
                  'p.Characteristic-p.Goal': '#a60b97',
                  'p.Time-p.Goal': '#97567c',
                  'p.Explanation-p.Time': '#90f72f',
                  'p.Instrument-p.Manner': '#2b1869',
                  'p.Possession-p.Ancillary': '#a9672c',
                  'p.Instrument-p.Instrument': '#6eb1ef',
                  'p.Ensemble-p.Ancillary': '#93fb41',
                  'p.Recipient-p.Gestalt': '#0674a2',
                  'p.Agent-p.Source': '#bf427f',
                  'p.Whole-p.Source': '#dae5cb',
                  'p.Stimulus-p.Explanation': '#108bd6',
                  'p.Stimulus-p.Direction': '#aa0f64',
                  'p.ComparisonRef-p.Purpose': '#65fb63',
                  'p.ComparisonRef-p.Locus': '#e48da2',
                  'p.Theme-p.Ancillary': '#685b19',
                  'p.Identity-p.ComparisonRef': '#caac20',
                  'p.QuantityItem-p.Stuff': '#a1f649',
                  'p.Recipient-p.Direction': '#a8ba9d',
                  'p.Path-p.Locus': '#03c408',
                  'p.Originator-p.Agent': '#b46878',
                  'p.Beneficiary-p.Gestalt': '#26eaf0',
                  'p.Possessor-p.Ancillary': '#dd8d5e',
                  'p.Beneficiary-p.Goal': '#212bd7',
                  'p.OrgMember-p.PartPortion': '#bd7620',
                  'p.PartPortion-p.ComparisonRef': '#6fd197',
                  'p.Frequency-p.Extent': '#8a9e22',
                  'p.Beneficiary-p.Direction': '#094599',
                  'p.Characteristic-p.Stuff': '#02889c',
                  'p.Manner-p.Extent': '#686d06',
                  'p.Cost-p.Cost': '#f4b518',
                  'p.Theme-p.Whole': '#5a51fb',
                  'p.Frequency-p.Frequency': '#d26bc7',
                  'p.Purpose-p.Locus': '#80e1ac',
                  'p.Force-p.Gestalt': '#1063d3',
                  'p.Characteristic-p.Ancillary': '#947622',
                  'p.ComparisonRef-p.Source': '#b0954c',
                  'p.Org-p.Instrument': '#e2bfce',
                  'p.Theme-p.Characteristic': '#44b67f',
                  'p.Characteristic-p.Topic': '#b90264',
                  'p.Locus-p.Goal': '#5d62c0',
                  'p.Locus-p.Whole': '#e4222b',
                  'p.Theme-p.Locus': '#60211c',
                  'p.Frequency-p.Manner': '#6b5831',
                  'p.Locus-p.Ancillary': '#8de37d',
                  'p.Topic-p.Identity': '#10a385',
                  'p.Org-p.Goal': '#b42090',
                  'p.SetIteration-p.SetIteration': '#11e7a6',
                  'p.PartPortion-p.Goal': '#ee8159',
                  'p.ComparisonRef-p.Ancillary': '#3270a9',
                  'p.Force-p.Force': '#dc6a3a',
                  'p.Approximator-p.Extent': '#005d48',
                  'p.Manner-p.Stuff': '#920903',
                  'p.Path-p.Goal': '#543e80',
                  'p.Explanation-p.Source': '#e65656',
                  'p.Topic-p.Goal': '#31bcfc',
                  'p.Possession-p.Locus': '#1312e3',
                  'p.Circumstance-p.Path': '#8b9109',
                  'p.Gestalt-p.Source': '#7050ae',
                  'p.Agent-p.Locus': '#c9846e',
                  'p.Stimulus-p.Source': '#180a5f',
                  'p.Org-p.Whole': '#2a3053',
                  'p.Org-p.Source': '#ad1e85',
                  'p.Time-p.Extent': '#b1d4fa',
                  'p.Possessor-p.Locus': '#ae306d',
                  'p.Force-p.Source': '#727a29',
                  'p.Gestalt-p.Topic': '#f47f98',
                  'p.Cost-p.Manner': '#a61141',
                  'p.Means-p.Path': '#54d11a',
                  'p.Originator-p.Instrument': '#44fe8a',
                  'p.PartPortion-p.Instrument': '#4f7170',
                  'p.Possession-p.Possession': '#d3abe4',
                  'p.Agent-p.Beneficiary': '#1c515e',
                  'p.Instrument-p.Locus': '#4460b0',
                  'p.Instrument-p.Theme': '#1bed0b',
                  'p.Duration-p.Gestalt': '#2f787f',
                  'p.Path-p.Path': '#3637c0',
                  'p.Theme-p.Source': '#54a6f9',
                  'p.Time-p.Gestalt': '#24ff12',
                  'p.Time-p.Direction': '#9e135c',
                  'p.Goal-p.Whole': '#5fad91',
                  'p.Explanation-p.Manner': '#983754',
                  'p.Time-p.Interval': '#5cc4a8',
                  'p.Org-p.Locus': '#434851',
                  'p.Gestalt-p.Purpose': '#9ff474',
                  'p.Stimulus-p.Theme': '#12dfa1',
                  'p.Locus-p.Gestalt': '#636042',
                  'p.Extent-p.Identity': '#1414fd',
                  'p.ComparisonRef-p.Beneficiary': '#f47ef3',
                  'p.Experiencer-p.Agent': '#21883e',
                  'p.Time-p.Duration': '#98b42b',
                  'p.SocialRel-p.Source': '#4f3f8f',
                  'p.Whole-p.Circumstance': '#c70411',
                  'p.Purpose-p.Goal': '#f2f199'}
        # Load once per call (ZeroGPU session). If you want to cache between calls, use a global
        # lazy-holder that only stores CPU objects; but safest with ZeroGPU is to init here.
        model_name = "WesScivetti/SNACS_Multilingual"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else None)
        # Build ONE pipeline (your custom one) that already computes probabilities
        token_classifier = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",   # get grouped spans
            pipeline_class=MyPipeline,
            device=0  # important for ZeroGPU
        )

        # Run once
        results_with_probs = token_classifier(text)
        # If you also want the vanilla aggregated view, you can derive it from the same results:
        results_simple = [
            {
                "word": e.get("word"),
                # MyPipeline returns grouped entities with 'entity_group'
                "entity_group": e.get("entity_group", e.get("entity")),
                "start": e["start"],
                "end": e["end"],
                "score": e["score"],
            }
            for e in results_with_probs
        ]

        # ---------- your rendering (with one small bug fix: use last_idx2) ----------
        #color_dict = {...}  # keep your dict as-is

        sorted_results = sorted(results_simple, key=lambda x: x["start"])
        sorted_results2 = sorted(results_with_probs, key=lambda x: x["start"])

        # FIRST VIEW
        output = ""
        last_idx = 0
        for entity in sorted_results:
            start = entity["start"]; end = entity["end"]
            label = entity["entity_group"]; score = entity["score"]
            word = html.escape(text[start:end])
            output += html.escape(text[last_idx:start])
            color = color_dict.get(label, "#D3D3D3")
            tooltip = f"{label} ({score:.2f})"
            word_with_label = f"{word}_{label}"
            output += (
                f"<span style='background-color: {color}; padding: 2px; border-radius: 4px;' "
                f"title='{tooltip}'>{word_with_label}</span>"
            )
            last_idx = end
        output += html.escape(text[last_idx:])

        # SECOND VIEW (top-5)
        output2 = ""
        last_idx2 = 0   # FIX: track the second stream separately
        for entity in sorted_results2:
            start = entity["start"]; end = entity["end"]
            label = entity.get("entity_group", entity.get("entity"))
            probabilities = entity["probabilities"]
            word = html.escape(text[start:end])
            output2 += html.escape(text[last_idx2:start])
            color = color_dict.get(label, "#D3D3D3")

            top5 = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top5_lines = [f"{html.escape(k)}: {v:.2%}" for k, v in top5]
            tooltip = "Top-5&#10;" + "&#10;".join(top5_lines)

            word_with_label = f"{word}_{label}"
            output2 += (
                f"<span style='background-color: {color}; padding: 2px; border-radius: 4px;' "
                f"title='{tooltip}'>{word_with_label}</span>"
            )
            last_idx2 = end
        output2 += html.escape(text[last_idx2:])

        # TABLE
        table_html = "<table style='border-collapse: collapse; font-family: sans-serif;'>"
        table_html += "<tr><th style='border: 1px solid #ccc; padding: 6px;'>Token</th>"
        table_html += "<th style='border: 1px solid #ccc; padding: 6px;'>SNACS Label</th>"
        table_html += "<th style='border: 1px solid #ccc; padding: 6px;'>Confidence</th></tr>"
        for e in sorted_results:
            token = html.escape(e["word"])
            label = e["entity_group"]
            score = f"{e['score']:.2f}"
            color = color_dict.get(label, "#D3D3D3")
            table_html += "<tr>"
            table_html += f"<td style='border: 1px solid #ccc; padding: 6px; background-color: {color};'>{token}</td>"
            table_html += f"<td style='border: 1px solid #ccc; padding: 6px; background-color: {color};'>{label}</td>"
            table_html += f"<td style='border: 1px solid #ccc; padding: 6px;'>{score}</td>"
            table_html += "</tr>"
        table_html += "</table>"

        styled_html = f"<div style='font-family: sans-serif; line-height: 1.6;'>{output}</div>"
        styled_html2 = f"<div style='font-family: sans-serif; line-height: 1.6;'>{output2}</div>"
        return styled_html, styled_html2, table_html
    except Exception as e:
        # Force the real error into the Space logs
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        # Also show something in the UI so you know it’s the worker, not Gradio
        return f"<pre>{html.escape(repr(e))}</pre>", "", ""

iface = gr.Interface(
    fn=classify_tokens,
    inputs=gr.Textbox(lines=4, placeholder="Enter a sentence...", label="Input Text"),
    outputs=[
        gr.HTML(label="SNACS Tagged Sentence"),
        gr.HTML(label="SNACS Tagged Sentence with No Label Aggregation"),
        gr.HTML(label="SNACS Table with Colored Labels")
    ],
    title="SNACS Classification",
    description="SNACS Classification. Now Multilingual! See the <a href='https://arxiv.org/abs/1704.02134'>SNACS guidelines</a> for details.",
    theme="default"
)
iface.launch()
