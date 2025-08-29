import html
import gradio as gr
import spaces
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import torch
import numpy as np


# Load the pipeline (token classification)
# token_classifier = pipeline("token-classification", model="WesScivetti/SNACS_English", aggregation_strategy="simple")

@spaces.GPU  # <-- required for ZeroGPU
def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

@spaces.GPU  # <-- required for ZeroGPU
class MyPipeline(TokenClassificationPipeline):
    def postprocess(self, all_outputs, aggregation_strategy="none", ignore_labels=None):
        if ignore_labels is None:
            ignore_labels = ["O"]

        # Convenience
        id2label = self.model.config.id2label
        # Ensure deterministic ordering of labels in the probabilities dict
        label_ids_sorted = sorted(id2label.keys())
        labels_sorted = [id2label[i] for i in label_ids_sorted]

        def softmax(logits):
            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted = logits - maxes
            exp = np.exp(shifted)
            return exp / exp.sum(axis=-1, keepdims=True)

        # We'll keep ALL pre_entities from all chunks to compute probabilities for grouped spans later.
        all_pre_entities = []
        all_grouped_entities = []

        # Get map from the first output, it's the same for all chunks
        word_to_chars_map = all_outputs[0].get("word_to_chars_map")
        sentence = all_outputs[0]["sentence"]

        for model_outputs in all_outputs:
            # logits -> scores (probabilities per class)
            if self.framework == "pt" and model_outputs["logits"][0].dtype in (torch.bfloat16, torch.float16):
                logits = model_outputs["logits"][0].to(torch.float32).numpy()
            else:
                logits = model_outputs["logits"][0].numpy()

            scores = softmax(logits)

            input_ids = model_outputs["input_ids"][0]
            offset_mapping = (
                model_outputs["offset_mapping"][0] if model_outputs["offset_mapping"] is not None else None
            )
            special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()
            word_ids = model_outputs.get("word_ids")

            if self.framework == "tf":
                input_ids = input_ids.numpy()
                offset_mapping = offset_mapping.numpy() if offset_mapping is not None else None

            # Build pre-entities (per token), preserving per-class distributions
            pre_entities = self.gather_pre_entities(
                sentence,
                input_ids,
                scores,
                offset_mapping,
                special_tokens_mask,
                aggregation_strategy,
                word_ids=word_ids,
                word_to_chars_map=word_to_chars_map,
            )

            # Aggregate into entities according to the chosen strategy
            grouped_entities = self.aggregate(pre_entities, aggregation_strategy)

            # Filter ignore labels
            grouped_entities = [
                e for e in grouped_entities
                if e.get("entity", None) not in ignore_labels
                   and e.get("entity_group", None) not in ignore_labels
            ]

            all_pre_entities.extend(pre_entities)
            all_grouped_entities.extend(grouped_entities)

        # If there were multiple chunks, reconcile overlapping entities as before.
        num_chunks = len(all_outputs)
        if num_chunks > 1:
            all_grouped_entities = self.aggregate_overlapping_entities(all_grouped_entities)

        # ---- Attach probabilities to each entity ----
        #
        # Strategy:
        #  - For token-level outputs (aggregation_strategy == NONE): we can map directly by token index.
        #  - For span/grouped outputs (SIMPLE/FIRST/MAX/AVERAGE): average the per-class distributions of the
        #    tokens that overlap the entity's [start, end), and (when possible) whose argmax tag matches the
        #    entity tag. If offsets are unavailable, we fallback to "best effort" using the entity word length.
        #

        def token_pred_label_id(token_scores: np.ndarray) -> int:
            return int(token_scores.argmax())

        def label_from_entity_dict(ent: dict) -> str:
            # entity_group for grouped results, entity for NONE
            if "entity_group" in ent and ent["entity_group"] is not None:
                return ent["entity_group"]
            if "entity" in ent and ent["entity"] is not None:
                # Strip B-/I- for comparison purposes
                tag = ent["entity"]
                if tag.startswith("B-") or tag.startswith("I-"):
                    return tag[2:]
                return tag
            return None

        def spans_overlap(a_start, a_end, b_start, b_end):
            if a_start is None or a_end is None or b_start is None or b_end is None:
                return False
            return max(a_start, b_start) < min(a_end, b_end)

        # Precompute convenience arrays from pre_entities
        pre_tokens = []
        for pe in all_pre_entities:
            # Each pre-entity has: word, scores (np.array), start, end, index, is_subword
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
            arr = np.stack([t["scores"] for t in token_list], axis=0)  # (T, C)
            avg = np.nanmean(arr, axis=0)
            # Ensure numeric stability (already normalized but just in case)
            s = float(avg.sum())
            if s > 0:
                avg = avg / s
            return avg

        results_with_probs = []
        for ent in all_grouped_entities:
            ent_start = ent.get("start")
            ent_end = ent.get("end")
            ent_tag = label_from_entity_dict(ent)

            candidate_tokens = []

            if aggregation_strategy == "none":
                # Token-level; match by index
                idx = ent.get("index")
                if idx is not None:
                    for t in pre_tokens:
                        if t["index"] == idx:
                            candidate_tokens = [t]
                            break
            else:
                # Span-level; collect tokens that overlap the span
                overlapping = [t for t in pre_tokens if spans_overlap(ent_start, ent_end, t["start"], t["end"])]
                if ent_tag is not None:
                    # Match tag ignoring B-/I- prefixes on token labels
                    def strip_bi(lbl):
                        return lbl[2:] if lbl.startswith("B-") or lbl.startswith("I-") else lbl

                    overlapping = [t for t in overlapping if strip_bi(t["pred_label"]) == ent_tag]
                candidate_tokens = overlapping

            avg = average_probs(candidate_tokens)

            if avg is None:
                # Fallback: if we somehow couldn't compute, create a degenerate distribution
                # focusing on the predicted class in the entity
                probs_vec = np.zeros((len(labels_sorted),), dtype=float)
                # Try to infer entity's class index
                if "entity" in ent and ent["entity"] is not None:
                    ent_label = ent["entity"]
                elif "entity_group" in ent and ent["entity_group"] is not None:
                    ent_label = ent["entity_group"]
                else:
                    ent_label = None

                # Strip B-/I- if present
                if ent_label is not None and (ent_label.startswith("B-") or ent_label.startswith("I-")):
                    ent_label = ent_label[2:]

                # Find first matching label id (allow both "B-"/"I-" and plain)
                chosen_i = None
                for i, lab in enumerate(labels_sorted):
                    base = lab[2:] if lab.startswith(("B-", "I-")) else lab
                    if ent_label == base:
                        chosen_i = i
                        break
                if chosen_i is None:
                    chosen_i = 0
                probs_vec[chosen_i] = 1.0
            else:
                # avg is class-ordered by model's original id order; map to labels_sorted
                # Build a lookup from original id -> position in labels_sorted
                # (labels_sorted was built in id order, so this is already aligned)
                probs_vec = avg

            # Attach probabilities as a readable dict
            ent["probabilities"] = {labels_sorted[i]: float(probs_vec[i]) for i in range(len(labels_sorted))}
            results_with_probs.append(ent)

        return results_with_probs


@spaces.GPU  # <-- required for ZeroGPU
def classify_tokens(text):
    #
    # def load_token_clf(model_name_or_path, device=None):
    #     tok = AutoTokenizer.from_pretrained(model_name_or_path)
    #     mdl = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
    #     if device is None:
    #         device = 0 if torch.cuda.is_available() else -1
    #     if device >= 0:
    #         mdl = mdl.to(device)
    #     return tok, mdl
    #
    # @torch.no_grad()
    # def tokens_with_distributions(tokenizer, model, text, *, truncation=True):
    #     """
    #     Returns a list of dicts, one per *subword token* (special tokens excluded):
    #       {
    #         'token': str,              # the tokenizer piece (e.g., '▁Montréal' or '##n')
    #         'text': str,               # exact substring from the original text
    #         'start': int, 'end': int,  # char offsets into `text`
    #         'probs': {label: float, ...},   # full distribution over all labels
    #         'top_label': str,          # argmax label (convenience)
    #         'top_score': float         # argmax prob (convenience)
    #       }
    #     """
    #     enc = tokenizer(
    #         text,
    #         return_tensors="pt",
    #         return_offsets_mapping=True,
    #         truncation=truncation
    #     )
    #     offset_mapping = enc.pop("offset_mapping")[0]  # (seq_len, 2)
    #     input_ids = enc["input_ids"][0]
    #     # Move tensors to model device
    #     enc = {k: v.to(model.device) for k, v in enc.items()}
    #
    #     logits = model(**enc).logits[0]  # (seq_len, num_labels)
    #     probs = torch.softmax(logits, dim=-1)
    #
    #     id2label = model.config.id2label
    #     # Ensure indexable by int
    #     if isinstance(id2label, dict):
    #         id2label = {int(k): v for k, v in id2label.items()}
    #
    #     out = []
    #     for i, (start, end) in enumerate(offset_mapping.tolist()):
    #         # Skip special tokens that have (0, 0) or otherwise map to no chars
    #         if start == end:
    #             continue
    #
    #         token_str = tokenizer.convert_ids_to_tokens(int(input_ids[i]))
    #         # Full distribution over original labels (e.g., B-PER/I-PER/O …)
    #         dist = {id2label[j]: float(probs[i, j]) for j in range(probs.shape[-1])}
    #         # Argmax for convenience
    #         top_j = int(torch.argmax(probs[i]).item())
    #         top_label = id2label[top_j]
    #         top_score = float(probs[i, top_j])
    #
    #         out.append({
    #             "token": token_str,
    #             "text": text[start:end],
    #             "start": start,
    #             "end": end,
    #             "probs": dist,
    #             "top_label": top_label,
    #             "top_score": top_score,
    #         })
    #     return out

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

    # get the probability distributions
    # tok_sn, mdl_sn = load_token_clf("WesScivetti/SNACS_Multilingual")
    # results2 = tokens_with_distributions(tok_sn, mdl_sn, text)
    # print(results2, file=sys.stderr)
    # sorted_results2 = sorted(results2, key=lambda x: x["start"])

    token_classifier = pipeline("token-classification", model="WesScivetti/SNACS_Multilingual",
                                aggregation_strategy="simple")

    token_classifier2 = pipeline("token-classification", model="WesScivetti/SNACS_Multilingual",
                                 pipeline_class=MyPipeline)

    results = token_classifier(text)
    results2 = token_classifier2(text)

    print(results)
    print(results2)

    sorted_results = sorted(results, key=lambda x: x["start"])
    sorted_results2 = sorted(results2, key=lambda x: x["start"])
    output = ""
    last_idx = 0

    for entity in sorted_results:
        start = entity["start"]
        end = entity["end"]
        label = entity["entity_group"]
        score = entity["score"]
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

    output2 = ""
    last_idx2 = 0
    for entity in sorted_results2:
        start = entity["start"]
        end = entity["end"]
        label = entity["entity"]
        score = entity["score"]
        probabilities = entity["probabilities"]
        word = html.escape(text[start:end])
        output2 += html.escape(text[last_idx:start])

        color = color_dict.get(label, "#D3D3D3")
        tooltip = f"{label} ({probabilities:.2f})"
        word_with_label = f"{word}_{label}"

        output2 += (
            f"<span style='background-color: {color}; padding: 2px; border-radius: 4px;' "
            f"title='{tooltip}'>{word_with_label}</span>"
        )

        last_idx = end
    output2 += html.escape(text[last_idx:])

    # for entity in sorted_results2:
    #     start = entity["start"]
    #     end = entity["end"]
    # label = entity["top_label"]
    # score = entity["top_score"]
    # dist = entity["probs"]
    # word = html.escape(text[start:end])
    # output += html.escape(text[last_idx:start])
    #
    # color = color_dict.get(label, "#D3D3D3")
    # tooltip = f"{label} ({score:.2f})"
    # word_with_label = f"{word}_{label}"
    #
    # output += (
    #     f"<span style='background-color: {color}; padding: 2px; border-radius: 4px;' "
    #     f"title='{tooltip}'>{word_with_label}</span>"
    # )
    #
    # last_idx = end
    # output += html.escape(text[last_idx:])

    table = [
        [entity["word"], entity["entity_group"], f"{entity['score']:.2f}"]
        for entity in sorted_results
    ]

    # Return both: HTML and table
    styled_html = f"<div style='font-family: sans-serif; line-height: 1.6;'>{output}</div>"

    styled_html2 = f"<div style='font-family: sans-serif; line-height: 1.6;'>{output2}</div>"

    # Generate a colored HTML table
    table_html = "<table style='border-collapse: collapse; font-family: sans-serif;'>"
    table_html += "<tr><th style='border: 1px solid #ccc; padding: 6px;'>Token</th>"
    table_html += "<th style='border: 1px solid #ccc; padding: 6px;'>SNACS Label</th>"
    table_html += "<th style='border: 1px solid #ccc; padding: 6px;'>Confidence</th></tr>"

    for entity in sorted_results:
        token = html.escape(entity["word"])
        label = entity["entity_group"]
        score = f"{entity['score']:.2f}"
        color = color_dict.get(label, "#D3D3D3")

        table_html += "<tr>"
        table_html += (
            f"<td style='border: 1px solid #ccc; padding: 6px; background-color: {color};'>{token}</td>"
        )
        table_html += (
            f"<td style='border: 1px solid #ccc; padding: 6px; background-color: {color};'>{label}</td>"
        )
        table_html += f"<td style='border: 1px solid #ccc; padding: 6px;'>{score}</td>"
        table_html += "</tr>"
    table_html += "</table>"

    return styled_html, styled_html2, table_html

# iface = gr.Interface(
#     fn=classify_tokens,
#     inputs=gr.Textbox(lines=4, placeholder="Enter a sentence...", label="Input Text"),
#     outputs=[
#         gr.HTML(label="SNACS Tagged Sentence"),
#          gr.HTML(label="SNACS Tagged Sentence with No Label Aggregation"),
#         gr.HTML(label="SNACS Table with Colored Labels")
#     ],
#     title="SNACS Classification",
#     description="SNACS Classification. Now Multilingual! See the <a href='https://arxiv.org/abs/1704.02134'>SNACS guidelines</a> for details.",
#     theme="default"
# )

# iface.launch()
