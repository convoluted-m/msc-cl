# Define a Tokeniser class with the following public methods

class Tokeniser:
    def __init__(self):
        pass
    # Method to tokenise text splitting it on whitespace (tab characters and newlines) and punctuation signs
    def tokenise_on_punctuation(self, text):
        tokens = text.split()

    def train(self, text):
        pass

    def tokenise(self, text, use_unk=False):
        pass

    def tokenise_with_count_threshold(self, text, threshold, use_unk=False):
        pass

    def tokenise_with_freq_threshold(self, text, threshold, use_unk=False):
        pass



# test 
test_text = "Lovely weather we are having!"

