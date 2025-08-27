import html
from collections import defaultdict

import gradio as gr
import spaces
from transformers import pipeline


# --- Helper: small CSS for nice tooltips on hover ---
HOVER_CSS = """
<style>
.hover-sentence { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; line-height: 1.8; }
.tok { position: relative; display: inline-block; }
.tok .tip {
  display: none; position: absolute; left: 0; top: 100%; margin-top: 6px;
  background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
  box-shadow: 0 8px 28px rgba(0,0,0,.15); z-index: 9999;
  max-height: 360px; overflow: auto; min-width: 320px; padding: 10px 12px;
  white-space: normal;
}
.tok:hover .tip { display: block; }
.tip h4 { margin: 0 0 6px 0; font-size: 13px; font-weight: 700; color: #111827; }
.tip .row { display: grid; grid-template-columns: 1fr auto 120px; align-items: center; gap: 10px; font-size: 12px; padding: 3px 0; }
.tip .lbl { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
.tip .pct { font-variant-numeric: tabular-nums; }
.tip .bar { height: 6px; background: #eef2f7; border-radius: 4px; position: relative; }
.tip .bar > span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; }
</style>
"""


@spaces.GPU  # <-- required for ZeroGPU
def classify_tokens(text: str):
    # --- Your original color palette (unchanged) ---
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

    # --- Build pipeline (inside function so it runs on GPU via ZeroGPU) ---
    token_classifier = pipeline(
        "token-classification",
        model="WesScivetti/SNACS_Multilingual",
        aggregation_strategy="simple"
    )

    # 1) Aggregated results for top label + spans (for coloring & table)
    agg = token_classifier(text)
    agg_sorted = sorted(agg, key=lambda x: x["start"])

    # 2) Try to get full distributions aligned to words
    dists_per_word = []
    try:
        # Preferred path: return_all_scores with aggregation
        agg_with_scores = token_classifier(text, return_all_scores=True)
        # Expect: each item has 'scores' with list of {entity_group, score}
        for item in agg_with_scores:
            scores_map = {}
            # Some transformers versions use 'entity_group', others 'entity'
            for sc in item.get("scores", []):
                raw_label = sc.get("entity_group") or sc.get("entity")
                if raw_label is None:
                    continue
                # Normalize B-/I- prefixes if present
                if raw_label.startswith("B-") or raw_label.startswith("I-"):
                    raw_label = raw_label.split("-", 1)[1]
                scores_map[raw_label] = float(sc["score"])
            # Normalize to sum=1 (some versions already do; this is safe)
            total = sum(scores_map.values()) or 1.0
            for k in list(scores_map.keys()):
                scores_map[k] = scores_map[k] / total
            dists_per_word.append({
                "start": item["start"], "end": item["end"], "word": item["word"],
                "scores": scores_map
            })
    except Exception:
        # Fallback: compute distributions by averaging sub-token probabilities
        raw_tc = pipeline(
            "token-classification",
            model=token_classifier.model,
            tokenizer=token_classifier.tokenizer,
            aggregation_strategy=None
        )
        raw = raw_tc(text, return_all_scores=True)
        # Group sub-tokens into aggregated spans
        for span in agg_sorted:
            s0, s1 = span["start"], span["end"]
            buckets = [t for t in raw if t["start"] >= s0 and t["end"] <= s1]
            scores_sum = defaultdict(float)
            count = 0
            for t in buckets:
                for sc in t.get("scores", []):
                    raw_label = sc.get("entity_group") or sc.get("entity")
                    if raw_label is None:
                        continue
                    if raw_label.startswith("B-") or raw_label.startswith("I-"):
                        raw_label = raw_label.split("-", 1)[1]
                    scores_sum[raw_label] += float(sc["score"])
                count += 1
            if count == 0:
                # No subtokens? fall back to top label only
                scores_map = {span["entity_group"]: 1.0}
            else:
                # Average over subtokens, then normalize
                for k in list(scores_sum.keys()):
                    scores_sum[k] = scores_sum[k] / count
                total = sum(scores_sum.values()) or 1.0
                scores_map = {k: v / total for k, v in scores_sum.items()}
            dists_per_word.append({
                "start": s0, "end": s1, "word": span["word"], "scores": scores_map
            })

    # --- Output 1: your original tagged sentence (kept) ---
    output_html = ""
    last_idx = 0
    for ent in agg_sorted:
        start, end = ent["start"], ent["end"]
        label, score = ent["entity_group"], ent["score"]
        word = html.escape(text[start:end])
        output_html += html.escape(text[last_idx:start])
        color = color_dict.get(label, "#D3D3D3")
        tooltip = f"{label} ({score:.2f})"
        word_with_label = f"{word}_{label}"
        output_html += (
            f"<span style='background-color:{color}; padding:2px; border-radius:4px;' "
            f"title='{tooltip}'>{word_with_label}</span>"
        )
        last_idx = end
    output_html += html.escape(text[last_idx:])
    styled_html = f"<div style='font-family: sans-serif; line-height:1.6;'>{output_html}</div>"

    # --- Output 2: your original colored table (kept) ---
    table_html = "<table style='border-collapse: collapse; font-family: sans-serif;'>"
    table_html += "<tr><th style='border:1px solid #ccc; padding:6px;'>Token</th>"
    table_html += "<th style='border:1px solid #ccc; padding:6px;'>SNACS Label</th>"
    table_html += "<th style='border:1px solid #ccc; padding:6px;'>Confidence</th></tr>"
    for ent in agg_sorted:
        token = html.escape(ent["word"])
        label = ent["entity_group"]
        score = f"{ent['score']:.2f}"
        color = color_dict.get(label, "#D3D3D3")
        table_html += "<tr>"
        table_html += f"<td style='border:1px solid #ccc; padding:6px; background-color:{color};'>{token}</td>"
        table_html += f"<td style='border:1px solid #ccc; padding:6px; background-color:{color};'>{label}</td>"
        table_html += f"<td style='border:1px solid #ccc; padding:6px;'>{score}</td>"
        table_html += "</tr>"
    table_html += "</table>"

    # --- Output 3: NEW hoverable full-sentence with per-token distributions ---
    # Build a quick index from span (start,end) -> scores map
    span_to_scores = {}
    for d in dists_per_word:
        span_to_scores[(d["start"], d["end"])] = d["scores"]

    hover_parts = [HOVER_CSS, "<div class='hover-sentence'>"]
    last_idx = 0
    for ent in agg_sorted:
        s, e = ent["start"], ent["end"]
        label = ent["entity_group"]
        word = html.escape(text[s:e])
        color = color_dict.get(label, "#D3D3D3")
        hover_parts.append(html.escape(text[last_idx:s]))  # inter-token text

        # Prepare tooltip content (all labels, sorted by prob desc)
        scores_map = span_to_scores.get((s, e), {label: 1.0})
        sorted_labels = sorted(scores_map.items(), key=lambda kv: kv[1], reverse=True)

        # Build the tooltip body
        tip_rows = []
        for lbl, prob in sorted_labels:
            lbl_esc = html.escape(lbl)
            pct = f"{prob * 100:.1f}%"
            bar_w = f"{prob * 100:.1f}%"
            bar_color = color_dict.get(lbl, "#9ca3af")
            tip_rows.append(
                f"<div class='row'>"
                f"<div class='lbl' style='color:{bar_color}'>{lbl_esc}</div>"
                f"<div class='pct'>{pct}</div>"
                f"<div class='bar'><span style='background:{bar_color}; width:{bar_w};'></span></div>"
                f"</div>"
            )
        tip_html = (
            f"<div class='tip'>"
            f"<h4>Label probabilities</h4>"
            + "".join(tip_rows) +
            f"</div>"
        )

        # Visible token span (colored by top label)
        hover_parts.append(
            f"<span class='tok' "
            f"style='background:{color}; padding:2px 3px; border-radius:4px;'>"
            f"{word}{tip_html}</span>"
        )
        last_idx = e
    hover_parts.append(html.escape(text[last_idx:]))
    hover_parts.append("</div>")
    hover_html = "".join(hover_parts)

    return styled_html, table_html, hover_html


iface = gr.Interface(
    fn=classify_tokens,
    inputs=gr.Textbox(lines=4, placeholder="Enter a sentence...", label="Input Text"),
    outputs=[
        gr.HTML(label="SNACS Tagged Sentence"),
        gr.HTML(label="SNACS Table with Colored Labels"),
        gr.HTML(label="Hover sentence: per-token label distribution"),  # NEW
    ],
    title="SNACS Classification",
    description=(
        "SNACS Classification. Now Multilingual! "
        "See the <a href='https://arxiv.org/abs/1704.02134'>SNACS guidelines</a> for details."
    ),
    theme="default",
)

iface.launch()
