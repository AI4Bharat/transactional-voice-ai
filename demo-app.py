import argparse
import os
from datetime import datetime

import gradio as gr
import shortuuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from pipeline import PredictionPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--port", default=7860, type=int)
args = parser.parse_args()

GRADIO_PAGE_TITLE = "AI4Bharat: Tagged Speech Recognition Demo"
GRADIO_PAGE_DESC = """
<h1 style="text-align: center; margin-bottom: 1rem">AI4Bharat: Transactional Voice AI</h1>
"""

LOGGER_DB_PATH = "db/logger.tsv"
FEEDBACK_DB_PATH = "db/feedback.tsv"

AZURE_ACCCOUNT_URL = "https://classlm.blob.core.windows.net"
AZURE_CONTAINER = "backend-logs"
prediction_pipeline = PredictionPipeline()

language_mapping = {"English": "en", "Hindi": "hi"}


def upload_audio(fpath):
    default_credential = DefaultAzureCredential()

    blob_service_client = BlobServiceClient(
        AZURE_ACCCOUNT_URL, credential=default_credential
    )
    blob_client = blob_service_client.get_blob_client(
        container=AZURE_CONTAINER, blob=fpath.split("/")[-1]
    )

    with open(file=fpath, mode="rb") as data:
        blob_client.upload_blob(data)


def get_predictions(language, audio_file, mic_audio_file=None):
    if mic_audio_file is not None:
        file = mic_audio_file
    elif audio_file is not None:
        file = audio_file
    else:
        return "[Error - Audio File Not Provided]"

    uuid = shortuuid.uuid()
    new_file = file.rsplit("/", 1)[0]
    new_file = os.path.join(new_file, f"{uuid}.wav")
    os.rename(file, new_file)

    predictions = prediction_pipeline.predict(new_file, language_mapping[language])

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    row = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
        uuid,
        language,
        predictions["transcript_itn"],
        predictions["entities"],
        predictions["intent"],
        predictions["intent_orig"],
        predictions["intent_prob"],
        dt_string,
    )
    with open(LOGGER_DB_PATH, "a") as f:
        f.write(row)
    upload_audio(new_file)
    return (
        {"text": predictions["transcript_itn"], "entities": predictions["entities"]},
        predictions["transcript_itn"],
        predictions["intent"],
        predictions["entities"],
        uuid,
    )


def record_feedback(feedback, lang, transcript, entities, uuid):
    if not uuid:
        return
    if len(transcript) == 0 or len(entities) == 0:
        return
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    row = "{}\t{}\t{}\t{}\t{}\t{}\n".format(
        uuid, lang, transcript, entities, feedback, dt_string
    )
    with open(FEEDBACK_DB_PATH, "a") as f:
        f.write(row)


def record_feedback_correct(lang, transcript, entities, uuid):
    record_feedback("correct", lang, transcript, entities, uuid)


def record_feedback_incorrect(lang, transcript, entities, uuid):
    record_feedback("incorrect", lang, transcript, entities, uuid)


with gr.Blocks(title=GRADIO_PAGE_TITLE) as demo:
    with gr.Row():
        gr.HTML(value=GRADIO_PAGE_DESC)
    with gr.Row():
        with gr.Column():
            in_language = gr.Radio(
                label="Language", choices=["English", "Hindi"], value="English"
            )
            in_uploaded_audio = (
                gr.Audio(label="Upload Speech", source="upload", type="filepath"),
            )
            in_recorded_audio = (
                gr.Audio(label="Record Speech", source="microphone", type="filepath"),
            )
            btn_transcribe = gr.Button(value="Transcribe")
        with gr.Column():
            out_tagged_text = (
                gr.HighlightedText(label="Tagged Speech Recognition Output"),
            )
            out_transcript = (gr.Textbox(label="Speech Recognition Output", lines=1),)
            out_intent = (gr.Textbox(label="Intent Recognition Output", lines=1),)
            out_entities = (gr.Textbox(label="Entities JSON", lines=1, visible=True),)
            audio_uuid = (gr.Textbox(label="Sample UUID", lines=1),)
            with gr.Row():
                btn_feedback_correct = gr.Button("Correct")
                btn_feedback_incorrect = gr.Button("Incorect")
    btn_transcribe.click(
        fn=get_predictions,
        inputs=[in_language, in_uploaded_audio[0], in_recorded_audio[0]],
        outputs=[
            out_tagged_text[0],
            out_transcript[0],
            out_intent[0],
            out_entities[0],
            audio_uuid[0],
        ],
        api_name="transcribe",
    )
    btn_feedback_correct.click(
        fn=record_feedback_correct,
        inputs=[in_language, out_transcript[0], out_entities[0], audio_uuid[0],],
        api_name="feedback_correct",
    )
    btn_feedback_incorrect.click(
        fn=record_feedback_incorrect,
        inputs=[in_language, out_transcript[0], out_entities[0], audio_uuid[0],],
        api_name="feedback_incorrect",
    )

demo.launch(server_name="0.0.0.0", server_port=args.port)
