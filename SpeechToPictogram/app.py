import base64
import json
import random
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

def get_story(dream):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Im going to tell you of my dream and i want you to make a better more and more detailed story out of it so i can create a booklet with image generation. Can you split it into sections and put it inside of a json array section= nr of section, story= containing the story, alt_text= the alt text(make sure that the alt text is overall consistent and map each person in it to a known movie character):{dream}",
    temperature=0.7,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response["choices"][0]["text"]

def get_image(text):
    engine_id = "stable-diffusion-xl-beta-v2-2-2"
    api_host = "https://api.stability.ai"
    stability_key = os.getenv('stability_key')

    if stability_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {stability_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": f"a surrealist painting, inspired by Andrea Kowch, cg society contest winner, covered with cobwebs and dust, norman rockwell, detailed, {text}"
                }
            ],
            "cfg_scale": 25,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 50,
            "seed": 4294967295,
        },
    )

    if response.status_code != 200:
        raise Exception("Non-200 response: " + str(response.text))

    data = response.json()
    number = random.randint(0, 1000)
    with open(f"{number}.png", "wb") as f:
            f.write(base64.b64decode(data["artifacts"][0]["base64"]))

    return f"{number}.png"

import logging

def get_array(dream):
    json_start_index = dream.find("[")
        
    # Extract the JSON-formatted string from the original string
    json_string = dream[json_start_index:]

    # Parse the JSON-formatted string and convert it to a Python object
    my_object = json.loads(json_string)

    # Extract the JSON array from the Python object
    return my_object

def SpeechToText(audio, SelectedModel):
    logging.info('Loading model...')
    model = whisper.load_model(SelectedModel)

    logging.info('Loading audio...')
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    logging.info('Creating log-mel spectrogram...')
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    logging.info('Detecting language...')
    _, probs = model.detect_language(mel)
    lang = f"Language: {max(probs, key=probs.get)}"

    logging.info('Decoding audio to text...')
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    #text = get_story(result.text)
    #print(text)
    #text = get_array(text)
    #print(type(text))
    #car = []
    #for section in text:
    #   img = get_image(section["alt_text"])
    #    car.append(img)
    #    print('image added')
    car = [ "./585.png"]   
    text = "this is a test"

    return car, text


def clean_text(text):
    """
    we get rid of the commas and dots, maybe in the future there more things to get rid of in a sentence like !, ? ...
    Args:
        text (_type_): _description_
    Returns:
        _type_: _description_
    """
    print("cleaning text: ", text)
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
    print(cleaned_text)
    text_list = POS_tagging(concatString).make_predictions()
    if(len(text_list) < 1):
        return None
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
            carousel = gr.Gallery()
        btn1 = gr.Button("Transcribe")
        btn1.click(SpeechToText, inputs=[audio, dropdown], outputs=[carousel, transcript])
        with gr.Column():
            box = gr.Box()
            box.add(carousel.label(transcript.))  
  
    gr.Markdown("Made by [Omidreza](https://github.com/omidreza-amrollahi)")

        
demo.launch()
