import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        special_tokens = [self.pad_token, self.unk_token, 
                  self.bos_token, self.eos_token]

        # Word to id, id to word
        for index, token in enumerate(special_tokens):
            self.word_to_id[token] = index
            self.id_to_word[index] = token

        
        self.vocab_size = len(special_tokens)

        # Select distinct word
        words = set()
        for text in texts:
            for word in text.split():
                words.add(word)

        for word in sorted(words):
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = text.split()
        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
            for word in tokens
        ]
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = [
            self.id_to_word.get(index, self.unk_token)
            for index in ids
        ]
        return " ".join(words)
        pass
