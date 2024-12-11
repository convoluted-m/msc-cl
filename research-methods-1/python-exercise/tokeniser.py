# A text tokeniser and a basic corpus stats function

# Define a tokeniser class
class Tokeniser:
    '''A tokeniser class.'''
    def __init__(self):
        self.token_dict = {} # dictionary to store token counts
        self.is_trained = False #  to chck if the tokeniser has been trained
        self.total_token_count = 0

    # Tokenise text on whitespace & punctuation - without training
    def tokenise_on_punctuation(self, text):
        separators = "`!\"#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—`]"
        for separator in separators:
            text = text.replace(separator, " ") #replace separators with whitespace
        text = text.split() # split text on whitespace and put into a list
        return text
    
    # Train the tokeniser - prepare to use on a new text(extract counts and store in a dict)
    def train(self, text):
        tokens = self.tokenise_on_punctuation(text) # call the above tokenisation method to get tokens
        self.total_token_count = len(tokens)
        for i in tokens: # loop through tokens list and add counts to the dictionary
            if i not in self.token_dict:
                self.token_dict[i] = 1
            else:
                self.token_dict[i] += 1
        self.is_trained = True # set to True to check in the function below
        return self.token_dict
    
    # Tokenise after training
    def tokenise(self, text, use_unk=False):
        # should fail when  invoked without training
        if not self.is_trained: 
            raise RuntimeError("The tokeniser has not been trained yet.")
        else:
            output = []
            tokens = self.tokenise_on_punctuation(text)
            for token in tokens:
                if token in self.token_dict: # if token in vocab add to output
                    output.append(token) 
                else:                   # if token not in vocab
                    if use_unk == True:
                        output.append("UNK")  #replace with UNK
                    else:
                        output.extend(list(token)) # split into individual characters, add each individually
        return output # return tokenised output

    # Tokenise with a count threshold
    def tokenise_with_count_threshold(self, text, threshold, use_unk=False):
        if not self.is_trained: 
            raise RuntimeError("The tokeniser has not been trained yet.")
        else:
            tokens = self.tokenise_on_punctuation(text)
            output= []
            for token in tokens:
                if token in self.token_dict:
                    if self.token_dict[token] >= threshold:
                        output.append(token)
                    else:
                        if use_unk == True:
                            output.append("UNK") 
                        else:
                            output.extend(list(token))
                else:
                    if use_unk == True:
                        output.append("UNK") 
                    else:
                        output.extend(list(token))
        return output

    # Tokenise with a frequency threshold
    def tokenise_with_freq_threshold(self, text, threshold, use_unk=False):
        if not self.is_trained: 
            raise RuntimeError("The tokeniser has not been trained yet.")
        else:
            tokens = self.tokenise_on_punctuation(text)
            output= []
            for token in tokens:
                if token in self.token_dict:
                    if self.token_dict[token]/self.total_token_count >= 0:
                        output.append(token)
                    else:
                        if use_unk == True:
                            output.append("UNK") 
                        else:
                            output.extend(list(token))
                else:
                    if use_unk == True:
                        output.append("UNK") 
                    else:
                        output.extend(list(token))
        return output

# # Test the tokeniser class manually - to be deleted
# test = "Lovely weather we are having! Indeed they were having lovely weather."
# test_2 = "What a day today! What lovely weather!"
# tokeniser_instance = Tokeniser()
# print(tokeniser_instance.tokenise_on_punctuation(test))
# print(tokeniser_instance.train(test))
# print(tokeniser_instance.tokenise(test_2))
# print(tokeniser_instance.tokenise_with_count_threshold(test_2, 1))
# print(tokeniser_instance.tokenise_with_freq_threshold(test_2, 0.005))

# Define the get_stats function - return a dictionary with basic corpus statistics
def get_stats(text):
    token_dict = {}
    for i in text:
        if i not in token_dict:
            token_dict[i] = 1
        else:
            token_dict[i] += 1

    # number of tokens in the corpus - returns the number of keys which are unique tokens
    type_count = len(token_dict) 
    # total number of tokens in the corpus
    token_count = len(text) 
    # lexical variability
    type_token_ratio = type_count/token_count

    # token count by length
    token_length = []
    token_length_dict = {}
    for i in text: 
        token_length.append(len(i))

    for i in token_length:
        if i not in token_length_dict:
            token_length_dict[i] = 1
        else:
            token_length_dict[i] +=1

    token_count_by_length =  token_length_dict

    # average token length
    average_token_length = sum(token_length)/len(token_length)

    # token length - standard average
    squared_values_sum = 0
    for i in token_length:
        squared_values_sum += (i-average_token_length)**2
    token_length_std = (squared_values_sum/(len(token_length)-1))**1/2
    
    # Put all the variables into a stats dictionary 
    keys = ['type_count', 'token_count', 'type_token_ratio', 'token_count_by_length','average_token_length', 'token_length_std']
    values = [type_count, token_count, type_token_ratio, token_count_by_length, average_token_length, token_length_std]
    stats_dict= {keys[i]:values[i] for i in range(len(keys))}
    #stats_dict = dict(zip(keys, values))

    # print each stat in a separate line
    for key, value in stats_dict.items():
        print(f"{key}: {value}")

# # Test the stats function manually - to be deleted
# my_list = ['bla', 'orange', 'why']
# get_stats(my_list)