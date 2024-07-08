import dotenv
import os
import numpy 
from random import shuffle
import requests
from bs4 import BeautifulSoup
import dspy
from groq import Groq
from functools import cache

dotenv.load_dotenv()

MODEL_NAME = "mixtral-8x7b-32768"
client = Groq()

# build class prompt template
class BuildMessages:
    def __init__(self, system_prompt, user_prompt):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
    def render(self, **kwargs):
        sys = self.system_prompt.format(**kwargs)
        user = self.user_prompt.format(**kwargs)
        return [
            {"role":"system", "content":sys},
            {"role":"user", "content":user},
        ]

# define function translate using llm
@cache
def translate_grug(grug_text: str) -> str:
    sys_prompt = """Bạn là 1 chuyên gia trong việc dịch các văn bản với nhiều thứ tiếng khác nhau.
    Người dùng sẽ cung cấp văn bản được viết bởi ai đó có tên Grug và bạn sẽ cung cấp bản dịch."""

    user_prompt = """
    Dịch văn bản sau sang tiếng việt: {text}.
    Trả ra theo format: 
    Bản dịch: 

    Lưu ý: bạn chỉ được dùng tiếng việt. Không đưa ra các thông tin khác trong văn bản.
    """
    prompt_template = BuildMessages(
        system_prompt=sys_prompt,
        user_prompt=user_prompt
    )

    response = client.chat.completions.create(
        messages=prompt_template.render(text=grug_text),
        model=MODEL_NAME
    )
    return response.choices[0].message.content

# Prepare and process dataset
def prepare_dataset():
    PATH_DATA = "data/raw_data.txt"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(PATH_DATA):
        # download dataset
        res = requests.get("https://grugbrain.dev/")
        soup = BeautifulSoup(res.text, "html.parser")
        raw_text = [p.text for p in soup.find_all('p') if p.text]
        with open(PATH_DATA, mode='w') as file:
            for text in raw_text:
                file.write(text)
    else:
        raw_text = []
        with open(PATH_DATA, mode="r") as file:
            lines = file.readlines()
            for line in lines:
                raw_text.append(line.strip())

    # format dataset to prepare for dspy input
    dataset = []
    for grug_text in raw_text[:10]:
        translated = translate_grug(grug_text)
        dataset.append({"grug_text": grug_text, "plain_Vietnamese": translated})

    examples = []
    for row in dataset:
        examples.append(dspy.Example(grug_text=row['grug_text'],
                                     plain_Vietnamese=row['plain_Vietnamese']).with_inputs("plain_Vietnamese"))
        
    # train test split
    TEST_SIZE = 1/3.0
    shuffle(examples)
    train_size = int(len(examples) - TEST_SIZE * len(examples))
    return examples[:train_size], examples[train_size:] 
    

if __name__ == "__main__":
    train_dataset, test_dataset = prepare_dataset()
    print(train_dataset[0]['grug_text'])
    print(train_dataset[0]['plain_Vietnamese'])
