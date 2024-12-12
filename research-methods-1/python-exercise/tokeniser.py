# A text tokeniser and a basic corpus stats function

# Define a tokeniser class
class Tokeniser:
    '''A tokeniser class.'''
    def __init__(self):
        self.token_dict = {} # dictionary to store token counts
        self.is_trained = False #  will be used to check if the tokeniser has been trained
        self.total_token_count = 0


    # Tokenise text on whitespace & punctuation - without training
    def tokenise_on_punctuation(self, text):
        separators = "!\"#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—\n\t"
        for separator in separators: # loop through separators and replace with whitespace
            text = text.replace(separator, " ") 

        text = text.split() # split the text on whitespace and put it into a list
        return text
    

    # Train the tokeniser - prepare to use on a new text(extract counts and store in a dict)
    def train(self, text):
        tokens = self.tokenise_on_punctuation(text) # call the above tokenisation method to get the tokens list
        self.total_token_count = len(tokens)

        for i in tokens: # loop through the tokens list and add counts to the dictionary
            if i not in self.token_dict:
                self.token_dict[i] = 1
            else:
                self.token_dict[i] += 1

        self.is_trained = True # set to True to check in the method below    

    # Tokenise after training
    def tokenise(self, text, use_unk=False):
        # should fail when  invoked without training
        if not self.is_trained: 
            raise RuntimeError("The tokeniser has not been trained yet.")
        else:
            output = [] # create an empty list to store the output - vocab of tokens
            tokens = self.tokenise_on_punctuation(text)

            for token in tokens:
                if token in self.token_dict: # if token found in the vocab, add to the output
                    output.append(token) 
                else:                   # if token not in vocab
                    if use_unk == True: # if flag set to true, replace with 'UNK'
                        output.append("UNK")  
                    else:
                        output.extend(list(token)) # if false by default, split into individual characters, add each char individually
        return output # return tokenised output


    # Tokenise with a count threshold
    def tokenise_with_count_threshold(self, text, threshold, use_unk=False):
        if not self.is_trained: 
            raise RuntimeError("The tokeniser has not been trained yet.")
        else:
            tokens = self.tokenise_on_punctuation(text) # call the tokenisation method to get the tokens list
            output = [] # create an empty list to store the output
            
            for token in tokens: # loop through the tokens list
                try:
                    if self.token_dict[token] >= threshold: # if token count threshold times or more, add to the output
                            output.append(token)
                    else: # if token not found in the vocab, similarly, replace with 'UNK' or split into characters
                        if use_unk == True:
                            output.append("UNK") 
                        else:
                            output.extend(list(token))
                except: 
                    pass
        return output


    # Tokenise with a frequency threshold (relative frequency of the token in the training corpus)
    def tokenise_with_freq_threshold(self, text, threshold, use_unk=False):
        if not self.is_trained: 
            raise RuntimeError("The tokeniser has not been trained yet.")
        else:
            tokens = self.tokenise_on_punctuation(text)
            output= []

            for token in tokens:
                try:
                    if self.token_dict[token]/self.total_token_count >= threshold: 
                        output.append(token)
                    else:
                        if use_unk == True:
                            output.append("UNK") 
                        else:
                            output.extend(list(token))
                except:
                    pass

        return output


# Define the get_stats function - takes a tokenised text and returns a dictionary with the basic corpus stats
def get_stats(text):
    token_dict = {} # create an empty dictionary to store token counts
    for i in text: # loop through the tokens list and add counts to the dictionary
        if i not in token_dict:
            token_dict[i] = 1
        else:
            token_dict[i] += 1

    # Calculate the number of different tokens in the corpus - returns the number of keys which are unique tokens
    type_count = len(token_dict) 
    # Calculate the total number of tokens in the corpus
    token_count = len(text) 
    # Calculate lexical variability
    type_token_ratio = type_count/token_count

    # Calculate token count by length
    token_length = [] # create an empty list for token lengths
    token_count_by_length = {} # create an empty dictionary to store token counts by length
    for i in text: #  loop through the tokens list and add the length of each token to token_length list
        token_length.append(len(i))

    for i in token_length: # loop through token_length list and add counts to the dictionary
        if i not in token_count_by_length:
            token_count_by_length[i] = 1
        else:
            token_count_by_length[i] +=1

    # Calculate the average token length in the corpus
    average_token_length = sum(token_length)/len(token_length) 

    # Calculate the standard average length of tokens in the corpus
    squared_values_sum = 0
    for i in token_length:
        squared_values_sum += (i-average_token_length)**2
    token_length_std = (squared_values_sum/(len(token_length)-1))**1/2
    
    # Put all the variables into a stats dictionary 
    keys = ['type_count', 'token_count', 'type_token_ratio', 'token_count_by_length','average_token_length', 'token_length_std']
    values = [type_count, token_count, type_token_ratio, token_count_by_length, average_token_length, token_length_std]
    # stats_dict= {keys[i]:values[i] for i in range(len(keys))}
    stats_dict = dict(zip(keys, values))

    # print each stat in a separate line
    for key, value in stats_dict.items():
        print(f"{key}: {value}")