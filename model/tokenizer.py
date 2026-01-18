import os, io, sys
import sentencepiece as spm
from config import PROJECT_ROOT, vocab_size

class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
    
    def train(self, input_file, model_prefix="tokenizer"):
            
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            model_type="bpe",
            vocab_size=vocab_size,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            user_defined_symbols=['<|endoftext|>']
        )
        self.sp.Load(model_prefix + ".model")
    
    def load_weights(self, path):
        self.sp.Load(path)

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, ids):
        return self.sp.DecodeIds(ids)