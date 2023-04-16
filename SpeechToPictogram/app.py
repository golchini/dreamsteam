import whisper
import gradio as gr 
WhisperModels = ['tiny', 'base', 'small', 'medium', 'large']
import matplotlib.pyplot as plt
import matplotlib
import requests
matplotlib.use('AGG')
import io
from PIL import Image
import PIL
from io import BytesIO
import openai
import os
openai.organization = os.getenv('organization')
openai.api_key = os.getenv('api_key')

def SpeechToText(audio, SelectedModel):
    if audio == None : return "" 
    model = whisper.load_model(SelectedModel)
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the Max probability of language ?
    _, probs = model.detect_language(mel)
    lang = f"Language: {max(probs, key=probs.get)}"

    #  Decode audio to Text
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    return result.text, lang

def clean_text(text):
    """
    we get rid of the commas and dots, maybe in the future there more things to get rid of in a sentence like !, ? ...
    Args:
        text (_type_): _description_
    Returns:
        _type_: _description_
    """
    text = text.lower()
    text = text.replace(",", " ")
    text = text.replace(".", " ")
    text = text.replace("?", " ")
    text = text.replace("-", " ")
    text = text.split()
    new_string = []
    for temp in text:
        if temp:
            if temp == "i":
                temp = "I"
            new_string.append(temp)
    concatString = ' '.join(new_string)
    return new_string, concatString

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.data.path.append('/root/nltk_data')
from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

class POS_tagging():
    def __init__(self, concatString):
        self.concatString = concatString
    def handle_conjugation(self, tags):
        # here we do the conjugation for verbs
        new_sentence = []
        for index, item in enumerate(tags):
            if item[1] not in ['VBP', 'DT', 'IN', 'TO', 'VBG', 'VBD', 'VBN', 'VBZ']:
                new_sentence.append(item[0])
            elif item[1] in ['VBP', 'VBG', 'VBD', 'VBN', 'VBZ']:
                new_verb = WordNetLemmatizer().lemmatize(item[0],'v')
                if new_verb != "be":
                    new_sentence.append(new_verb)
        return new_sentence
    def make_predictions(self):
        tags = pos_tag(word_tokenize(self.concatString))
        return self.handle_conjugation(tags)

def generate_pic(text_to_search, ax):
    """
    we define a function here to use the api frpm arasaac, and return the image based on the text that we search
    ref: https://arasaac.org/developers/api
    Args:
        text_to_search (_type_): _description_
        ax (_type_): _description_
    """
    search_url = f"https://api.arasaac.org/api/pictograms/en/bestsearch/{text_to_search}"
    search_response = requests.get(search_url)
    search_json = search_response.json()
    if search_json:
        pic_url = f"https://api.arasaac.org/api/pictograms/{search_json[0]['_id']}?download=false"
        pic_response = requests.get(pic_url)
        img = Image.open(BytesIO(pic_response.content))
        ax.imshow(img)
        ax.set_title(text_to_search)
    else:
        try:
            response = openai.Image.create(
              prompt=text_to_search,
              n=2,
              size="512x512"
            )
            image_url = response['data'][0]['url']
            image_response = requests.get(image_url)
            img = Image.open(BytesIO(image_response.content))
            ax.imshow(img)
            ax.set_title(f"/{text_to_search}/")
        except:
            ax.set_title("Error!")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
# we generate an initial pictogram
# here we see the am, eating, an having a problem
def save_pictogram(transcript):
    """_summary_
    Args:
        text_list (_type_): _description_
    """
    cleaned_text, concatString = clean_text(transcript)
    text_list = POS_tagging(concatString).make_predictions()
    fig, ax = plt.subplots(1,len(text_list), figsize=(10,1.75))
    if len(text_list) > 1:
        for i, text in enumerate(text_list):
            generate_pic(text, ax[i])
    else:
        for i, text in enumerate(text_list):
            generate_pic(text, ax)
    fig.savefig("pictogram.png")
    safe_image = Image.open(r"pictogram.png")
    return safe_image

with gr.Blocks() as demo:
    gr.Markdown("# Speech to Pictogram App")
    gr.Markdown("The transcript of the audio can be in different languages, but the pictogram will only work for English")
    gr.Markdown("The pictures which are titled in slashes (/) are generated using stable diffusion, however, the API here has a limit of 25 calls per minute.")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(source="microphone", type="filepath")
            dropdown = gr.Dropdown(label="Whisper Model", choices=WhisperModels, value='base')
        with gr.Column():
            transcript = gr.Textbox(label="Transcript")
            lan = gr.Textbox(label="Language")
        btn1 = gr.Button("Transcribe")
        btn1.click(SpeechToText, inputs=[audio, dropdown], outputs=[transcript, lan])
    #if lan.value == "Language: en":
    image = gr.Image()
    btn2 = gr.Button("Generate Pictogram")
    btn2.click(save_pictogram, inputs=[transcript], outputs=[image])
    gr.Markdown("Made by [Omidreza](https://github.com/omidreza-amrollahi)")

        
demo.launch()
