from tokenizer import Tokenizer
from model import TarushGPTMini
from datasets import load_dataset
from config import num_megabytes

class Train:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = TarushGPTMini()

    def dataset(self):
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

        target_bytes = num_megabytes * 1024 * 1024
        current_bytes = 0

        with open("tarushgpt_data.txt", "w") as f:
            for item in self.dataset:
                text = item['text']
                f.write(text + '<|endoftext|>\n')
                current_bytes += len(text.encode('utf-8'))
                if current_bytes >= target_bytes:
                    break
        

    def train(self):
        self.dataset()
        with open("tarushgpt_data.txt", "r") as f:
            file = f.read()

        self.tokenizer.train("tarushgpt_data.txt")
        encoded = self.tokenizer.encode(file)
        