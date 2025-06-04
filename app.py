import html
import gradio as gr
import spaces
from transformers import pipeline



# Load the pipeline (token classification)
#token_classifier = pipeline("token-classification", model="WesScivetti/SNACS_English", aggregation_strategy="simple")


@spaces.GPU  # <-- required for ZeroGPU
def classify_tokens(text):

    color_dict = {'None': '#6adf97',
              'O': '#f18621',
              'B-p.Purpose-p.Purpose': '#554065',
              'B-p.SocialRel-p.Gestalt': '#8ea0d7',
              'B-p.Cost-p.Cost': '#f4b518',
              'B-p.Topic-p.Topic': '#976cae',
              'B-p.Originator-p.Gestalt': '#f25ca8',
              'B-p.Originator-p.Source': '#a08323',
              'B-p.Recipient-p.Goal': '#725be0',
              'B-p.Possessor-p.Possessor': '#b5ce9e',
              'p.Possessor-p.Possessor': '#b5ce9e',
              'B-p.Gestalt-p.Gestalt': '#34a8a9',
              'B-p.Ancillary-p.Ancillary': '#73f29f',
              'I-p.Ancillary-p.Ancillary': '#73f29f',
              'B-p.ComparisonRef-p.Goal': '#6a26db',
              'B-p.Source-p.Source': '#5cc334',
              'I-p.Source-p.Source': '#5cc334',
              'B-p.Theme-p.Theme': '#5b88c8',
              'B-p.Locus-p.Locus': '#4c39c8',
              'p.Locus-p.Locus': '#4c39c8',
              'B-p.Characteristic-p.Characteristic': '#661943',
              'B-p.Explanation-p.Explanation': '#852e58',
              'B-p.OrgMember-p.Possessor': '#e3bd42',
              'B-p.Goal-p.Goal': '#6bfc3c',
              'p.Goal-p.Goal': '#6bfc3c',
              'B-p.Manner-p.Manner': '#436097',
              'B-p.ComparisonRef-p.ComparisonRef': '#4df5a9',
              'B-p.Cost-p.Locus': '#fe5990',
              'B-p.Duration-p.Duration': '#5e454e',
              'B-p.Identity-p.Identity': '#cb49ed',
              'B-p.OrgMember-p.Gestalt': '#18fdd1',
              'B-p.Experiencer-p.Goal': '#400043',
              'B-p.QuantityItem-p.Whole': '#5f3ba4',
              'B-p.Whole-p.Gestalt': '#497114',
              'B-p.PartPortion-p.PartPortion': '#edfc14',
              'I-p.PartPortion-p.PartPortion': '#edfc14',
              'B-p.Time-p.Time': '#4605b0',
              'B-p.Approximator-p.Approximator': '#553ee1',
              'B-p.Direction-p.Direction': '#687447',
              'B-p.Locus-p.Direction': '#12b336',
              'B-p.Instrument-p.Path': '#0ccdda',
              'I-p.Instrument-p.Path': '#0ccdda',
              'B-p.QuantityItem-p.Gestalt': '#d88be2',
              'B-p.Species-p.Species': '#4dfc63',
              'B-p.Org-p.Ancillary': '#6a5b9c',
              'B-p.Agent-p.Gestalt': '#f373bf',
              'B-p.SocialRel-p.Ancillary': '#4ee1dc',
              'B-p.Circumstance-p.Locus': '#38abe5',
              'B-p.Circumstance-p.Circumstance': '#69caeb',
              'B-p.Path-p.Path': '#3637c0',
              'B-p.Whole-p.Whole': '#00d816',
              'I-p.Locus-p.Locus': '#4c39c8',
              'B-p.Manner-p.Locus': '#67fc5f',
              'I-p.Manner-p.Locus': '#67fc5f',
              'B-p.QuantityItem-p.QuantityItem': '#dbbc2d',
              'B-p.Theme-p.Purpose': '#cb56ba',
              'B-p.Goal-p.Locus': '#b3597f',
              'B-p.Extent-p.Extent': '#5cadfa',
              'I-p.Extent-p.Extent': '#5cadfa',
              'B-p.Experiencer-p.Gestalt': '#8275f4',
              'B-p.Means-p.Means': '#b1bfb7',
              'B-p.Beneficiary-p.Beneficiary': '#0e9582',
              'B-p.Org-p.Beneficiary': '#c48ea7',
              'B-p.Stimulus-p.Topic': '#a6af3a',
              'B-p.Recipient-p.Ancillary': '#a5ff4b',
              'B-p.Beneficiary-p.Possessor': '#c941dc',
              'B-p.Agent-p.Ancillary': '#d18ce9',
              'B-p.Theme-p.Gestalt': '#b71c4f',
              'B-p.StartTime-p.StartTime': '#9b3cf9',
              'I-p.ComparisonRef-p.ComparisonRef': '#4df5a9',
              'B-p.Cost-p.Extent': '#117f70',
              'B-p.Manner-p.Source': '#460233',
              'I-p.Manner-p.Source': '#460233',
              'B-p.Characteristic-p.Source': '#41c518',
              'I-p.Characteristic-p.Source': '#41c518',
              'B-p.Locus-p.Path': '#d3c136',
              'I-p.Topic-p.Topic': '#976cae',
              'B-p.Manner-p.ComparisonRef': '#32cbcb',
              'B-p.Extent-p.Whole': '#94454f',
              'I-p.Extent-p.Whole': '#94454f',
              'B-p.Experiencer-p.Beneficiary': '#1f2d98',
              'B-p.Theme-p.ComparisonRef': '#ef3f97',
              'I-p.Time-p.Time': '#4605b0',
              'B-p.Stuff-p.Stuff': '#9919e8',
              'B-p.Theme-p.Goal': '#d7c6d1',
              'B-p.Interval-p.Interval': '#042206',
              'B-p.Time-p.Whole': '#ecf0a1',
              'I-p.Circumstance-p.Circumstance': '#69caeb',
              'B-p.Stimulus-p.Beneficiary': '#af168a',
              'B-p.Time-p.Interval': '#5cc4a8',
              'B-p.Characteristic-p.Locus': '#ac54e6',
              'B-p.Characteristic-p.Extent': '#0ec04c',
              'B-p.EndTime-p.EndTime': '#29e89e',
              'B-p.Experiencer-p.Ancillary': '#bce155',
              'B-p.Agent-p.Agent': '#aac43b',
              'B-p.PartPortion-p.Source': '#9eb3c3',
              'B-p.Org-p.Locus': '#434851',
              'I-p.Characteristic-p.Locus': '#ac54e6',
              'B-p.Locus-p.Source': '#7121d7',
              'I-p.Locus-p.Source': '#7121d7',
              'B-p.Duration-p.Extent': '#ca1096',
              'B-p.Characteristic-p.Identity': '#345c8d',
              'B-p.Possession-p.PartPortion': '#e592aa',
              'B-p.Possession-p.Theme': '#a59bec',
              'B-p.Whole-p.Locus': '#0bc209',
              'B-p.Direction-p.Goal': '#9d90cd',
              'B-p.Gestalt-p.Locus': '#97f830',
              'B-p.Org-p.Gestalt': '#2f2c3c',
              'B-p.Stimulus-p.Goal': '#c40f02',
              'B-p.Theme-p.Instrument': '#a312ed',
              'B-p.Stimulus-p.Force': '#d98ddb',
              'I-p.Purpose-p.Purpose': '#554065',
              'B-p.Beneficiary-p.Theme': '#68fdb4',
              'B-p.Characteristic-p.Goal': '#a60b97',
              'I-p.Characteristic-p.Goal': '#a60b97',
              'B-p.Time-p.Goal': '#97567c',
              'I-p.Direction-p.Direction': '#687447',
              'B-p.Explanation-p.Time': '#90f72f',
              'B-p.Instrument-p.Manner': '#2b1869',
              'B-p.Possession-p.Ancillary': '#a9672c',
              'B-p.Instrument-p.Instrument': '#6eb1ef',
              'B-p.Ensemble-p.Ancillary': '#93fb41',
              'I-p.Cost-p.Locus': '#fe5990',
              'B-p.Recipient-p.Gestalt': '#0674a2',
              'B-p.Agent-p.Source': '#bf427f',
              'I-p.Circumstance-p.Locus': '#38abe5',
              'B-p.Whole-p.Source': '#dae5cb',
              'B-p.Stimulus-p.Explanation': '#108bd6',
              'B-p.Stimulus-p.Direction': '#aa0f64',
              'I-p.Explanation-p.Explanation': '#852e58',
              'I-p.Approximator-p.Approximator': '#553ee1',
              'B-p.ComparisonRef-p.Purpose': '#65fb63',
              'B-p.ComparisonRef-p.Locus': '#e48da2',
              'I-p.QuantityItem-p.Whole': '#5f3ba4',
              'B-p.Theme-p.Ancillary': '#685b19',
              'I-p.Manner-p.Manner': '#436097',
              'B-p.Identity-p.ComparisonRef': '#caac20',
              'I-p.Goal-p.Locus': '#b3597f',
              'B-p.QuantityItem-p.Stuff': '#a1f649',
              'B-p.Recipient-p.Direction': '#a8ba9d',
              'B-p.Path-p.Locus': '#03c408',
              'B-p.Originator-p.Agent': '#b46878',
              'B-p.Beneficiary-p.Gestalt': '#26eaf0',
              'B-p.Possessor-p.Ancillary': '#dd8d5e',
              'B-p.Beneficiary-p.Goal': '#212bd7',
              'B-p.OrgMember-p.PartPortion': '#bd7620',
              'B-p.PartPortion-p.ComparisonRef': '#6fd197',
              'B-p.Frequency-p.Extent': '#8a9e22',
              'B-p.Beneficiary-p.Direction': '#094599',
              'B-p.Characteristic-p.Stuff': '#02889c',
              'B-p.Manner-p.Extent': '#686d06',
              'I-p.Cost-p.Cost': '#f4b518',
              'B-p.Theme-p.Whole': '#5a51fb',
              'B-p.Frequency-p.Frequency': '#d26bc7',
              'B-p.Purpose-p.Locus': '#80e1ac',
              'B-p.Force-p.Gestalt': '#1063d3',
              'B-p.Characteristic-p.Ancillary': '#947622',
              'B-p.ComparisonRef-p.Source': '#b0954c',
              'B-p.Org-p.Instrument': '#e2bfce',
              'B-p.Theme-p.Characteristic': '#44b67f',
              'B-p.Characteristic-p.Topic': '#b90264',
              'I-p.Characteristic-p.Topic': '#b90264',
              'B-p.Locus-p.Goal': '#5d62c0',
              'B-p.Locus-p.Whole': '#e4222b',
              'B-p.Theme-p.Locus': '#60211c',
              'B-p.Frequency-p.Manner': '#6b5831',
              'I-p.Frequency-p.Manner': '#6b5831',
              'I-p.Ensemble-p.Ancillary': '#93fb41',
              'B-p.Locus-p.Ancillary': '#8de37d',
              'B-p.Topic-p.Identity': '#10a385',
              'B-p.Org-p.Goal': '#b42090',
              'B-p.SetIteration-p.SetIteration': '#11e7a6',
              'B-p.PartPortion-p.Goal': '#ee8159',
              'B-p.ComparisonRef-p.Ancillary': '#3270a9',
              'B-p.Force-p.Force': '#dc6a3a',
              'B-p.Approximator-p.Extent': '#005d48',
              'I-p.Duration-p.Duration': '#5e454e',
              'B-p.Manner-p.Stuff': '#920903',
              'B-p.Path-p.Goal': '#543e80',
              'B-p.Explanation-p.Source': '#e65656',
              'B-p.Topic-p.Goal': '#31bcfc',
              'I-p.Manner-p.ComparisonRef': '#32cbcb',
              'B-p.Possession-p.Locus': '#1312e3',
              'B-p.Circumstance-p.Path': '#8b9109',
              'B-p.Gestalt-p.Source': '#7050ae',
              'B-p.Agent-p.Locus': '#c9846e',
              'B-p.Stimulus-p.Source': '#180a5f',
              'B-p.Org-p.Whole': '#2a3053',
              'I-p.Locus-p.Direction': '#12b336',
              'B-p.Org-p.Source': '#ad1e85',
              'B-p.Time-p.Extent': '#b1d4fa',
              'I-p.Goal-p.Goal': '#6bfc3c',
              'B-p.Possessor-p.Locus': '#ae306d',
              'B-p.Force-p.Source': '#727a29',
              'B-p.Gestalt-p.Topic': '#f47f98',
              'I-p.Whole-p.Whole': '#00d816',
              'B-p.Cost-p.Manner': '#a61141',
              'B-p.Means-p.Path': '#54d11a',
              'B-p.Originator-p.Instrument': '#44fe8a',
              'B-p.PartPortion-p.Instrument': '#4f7170',
              'B-p.Possession-p.Possession': '#d3abe4',
              'I-p.Possession-p.Possession': '#d3abe4',
              'B-p.Agent-p.Beneficiary': '#1c515e',
              'B-p.Instrument-p.Locus': '#4460b0',
              'B-p.Instrument-p.Theme': '#1bed0b',
              'B-p.Duration-p.Gestalt': '#2f787f',
              'I-p.Path-p.Path': '#3637c0',
              'B-p.Theme-p.Source': '#54a6f9',
              'B-p.Time-p.Gestalt': '#24ff12',
              'B-p.Time-p.Direction': '#9e135c',
              'B-p.Goal-p.Whole': '#5fad91',
              'B-p.Explanation-p.Manner': '#983754',
              'I-p.Explanation-p.Manner': '#983754',
              'I-p.Time-p.Interval': '#5cc4a8',
              'I-p.Org-p.Locus': '#434851',
              'B-p.Gestalt-p.Purpose': '#9ff474',
              'B-p.Stimulus-p.Theme': '#12dfa1',
              'B-p.Locus-p.Gestalt': '#636042',
              'B-p.Extent-p.Identity': '#1414fd',
              'B-p.ComparisonRef-p.Beneficiary': '#f47ef3',
              'B-p.Experiencer-p.Agent': '#21883e',
              'B-p.Time-p.Duration': '#98b42b',
              'B-p.SocialRel-p.Source': '#4f3f8f',
              'B-p.Whole-p.Circumstance': '#c70411',
              'B-p.Purpose-p.Goal': '#f2f199'}

    token_classifier = pipeline("token-classification", model="WesScivetti/SNACS_English",
                                aggregation_strategy="simple")

    results = token_classifier(text)

    sorted_results = sorted(results, key=lambda x: x["start"])
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

        style_block = """
        <style>
          span:hover .tooltip {
            visibility: visible;
          }
        </style>
        """

        output += f"""
        <span style='position: relative; background-color: {color}; padding: 2px; border-radius: 4px; margin-right: 2px;'>
          {word}
          <span style='
            visibility: hidden;
            background-color: black;
            color: #fff;
            text-align: center;
            border-radius: 4px;
            padding: 2px 6px;
            position: absolute;
            z-index: 1;
            bottom: 120%;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
            font-size: 0.75rem;
          ' class='tooltip'>{label}</span>
        </span>
        """
        last_idx = end

    output += html.escape(text[last_idx:])

    style_block = """
    <style>
      span:hover .tooltip {
        visibility: visible;
      }
    </style>
    """

    return f"{style_block}<div style='font-family: sans-serif; line-height: 1.6;'>{output}</div>"


iface = gr.Interface(
    fn=classify_tokens,
    inputs=gr.Textbox(lines=4, placeholder="Enter a sentence...", label="Input Text"),
    outputs=gr.HTML(label="SNACS Tagged Sentence"),
    title="SNACS English Classification",
    description="SNACS English Classification. See the <a href='https://arxiv.org/abs/1704.02134'>SNACS guidelines</a> for details.",
    theme="default"
)

iface.launch()