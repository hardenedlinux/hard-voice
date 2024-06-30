"""-*-  indent-tabs-mode:nil; coding: utf-8 -*-.
 
Copyright (C) 2024
    HardenedLinux community
This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with this program.
If not, see <http://www.gnu.org/licenses/>.
 
"""

import gradio as gr
import soundfile as sf
import sounddevice as sd
import whisper

# Select from the following models: "tiny", "base", "small", "medium", "large"
model = whisper.load_model("small")

def record_voice():
    fs = 48000
    duration = 5
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write("my_Audio_file.flac", myrecording, fs)
    return "Recording done"

def transcribe():
    audio = "my_Audio_file.flac"
    options = {"fp16": False, "language": None, "task": "transcribe"}
    results = model.transcribe(audio, **options)
    return results["text"]

def translate():
    audio = "my_Audio_file.flac"
    options = {"fp16": False, "language": None, "task": "translate"}
    results = model.transcribe(audio, **options)
    return results["text"]

with gr.Blocks() as app:
    gr.Markdown("# Voice to Text")
    with gr.Row():
        record_button = gr.Button("Record")
        transcribe_button = gr.Button("Transcribe")
        translate_button = gr.Button("Translate")
    output_text = gr.Textbox(label="Output", interactive=False)

    record_button.click(record_voice, outputs=output_text)
    transcribe_button.click(transcribe, outputs=output_text)
    translate_button.click(translate, outputs=output_text)

app.launch()
