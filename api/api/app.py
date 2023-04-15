import os
import openai
import base64
import json
import requests


def save_story(text, folder, name):
    with open(f"./{folder}/{name}.txt", "w") as f:
        f.write(text)

def get_image(text, folder, name):
    engine_id = "stable-diffusion-xl-beta-v2-2-2"
    api_host = "https://api.stability.ai"
    api_key = "XXX"

    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
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
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, image in enumerate(data["artifacts"]):
        with open(f"./{folder}/{name}_{i}.png", "wb") as f:
            f.write(base64.b64decode(image["base64"]))

def get_story(dream):
    openai.api_key = "XXX"

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Im going to tell you of my dream and i want you to make a better more and more detailed story out of it so i can create a booklet with image generation. Can you split it into sections and put it inside of a json array section= nr of section, story= containing the story, alt_text= the alt text(make sure that the alt text is overall consistent and map each person in it to a known movie character):{dream}",
    temperature=0.7,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return json.loads(json.dumps(response))

def get_array(dream):
    json_start_index = dream.find("[")
        
    # Extract the JSON-formatted string from the original string
    json_string = dream[json_start_index:]

    # Parse the JSON-formatted string and convert it to a Python object
    my_object = json.loads(json_string)

    # Extract the JSON array from the Python object
    return my_object

dream_json = get_story("I was in a house with my mom and an other person. The person was a women relatively old. For some reasons, we did not like her and we tried to kill her. Before we could kill her, her two sons (big guys around 30 years old) ring at the house to look for her mom. I go upstairs in the house to hide the mom but the sons finds out and I run away from the house. Then, I keep run in the street and ask a car to take me and go very far. But the car get caught in the traffic, so I go out of the car and I keep hiding in the corners in the streets but the sons keep finding me everytime. At one point I go to a shop to hide but one of the guy finds me and I try to explain the shop owner that the guy will kill me but he does not trust me, so he gives me away to one of the sons. And in the meantime, one of the son almost beat my mom to death by kicking her.")
print(dream_json)
storyline = get_array(dream_json["choices"][0]["text"])
jibbie_id = dream_json["id"]


html_template = """
<!DOCTYPE html>
<html>
  <head>
    <title>My Booklet</title>
  </head>
  <body>
    <h1>My Booklet</h1>
    {}
  </body>
</html>
"""

div_template = """
<div>
  <img src="{}" alt="">
  <p>{}</p>
</div>
"""
divs = ""

for section in storyline:
    get_image(section["alt_text"], jibbie_id, section["section"])
    save_story(section["story"], jibbie_id, section["section"])
    divs += div_template.format(f"./{section['section']}_0.png", {section['story']})


final_html = html_template.format(divs)
with open(os.path.join(f"./{jibbie_id}", "booklet.html"), "w") as f:
    f.write(final_html)

print(final_html)