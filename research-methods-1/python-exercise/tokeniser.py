class Tokeniser:
    '''A tokeniser class'''
    def __init__(self):
        pass
    def tokenise_on_punctuation(self, text):
        separators = "`!\"#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—`]"
        for separator in separators:
            text = text.replace(separator, " ")
        text = text.split()
        return text

    def train(self, text):
        tokens = self.tokenise_on_punctuation(text)
        token_dict = {}
        for i in tokens:
            if i not in token_dict:
                token_dict[i] = 1
            else:
                token_dict[i] += 1
    
        return token_dict

    def tokenise(self, text, use_unk=False):
        pass

    def tokenise_with_count_threshold(self, text, threshold, use_unk=False):
        pass

    def tokenise_with_freq_threshold(self, text, threshold, use_unk=False):
        pass

sentence = "Lovely weather we are having!"

tokeniser_instance = Tokeniser()
print(tokeniser_instance.tokenise_on_punctuation(sentence))
print(tokeniser_instance.train(sentence))
