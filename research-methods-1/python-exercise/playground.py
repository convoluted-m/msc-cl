#text = "Lovely weather we are having!"
#tokens = text.split()

#Define pattern for tokenisation with regex
# import re
#tokens = re.findall(r'\w+', text) # \w alphanumeric character a-zA-Z and 0-9 and underscore; \s whtitespace
#tokens=re.split(" ", text) # tokenise on spaces
#print(tokens)
# 
#  FUNC 1 TOKENISE
#  def tokenise_on_punctuation(text):
#     separators = "`!\"#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—`]"
#     for separator in separators:
#         text = text.replace(separator, " ")
#     text = text.split()
#     return text

# sentence = "Lovely weather we are having!"
# print(tokenise_on_punctuation(sentence))

# FUNC 2 TRAIN
# sentence = "Lovely weather we are having!"

# def train(text):
#     separators = "`!\"#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—`]"
#     for separator in separators:
#         text = text.replace(separator, " ")
#     text = text.split()
#     token_dict = {}

#     for i in text:
#         if i not in token_dict:
#             token_dict[i] = 1
#         else:
#             token_dict[i] += 1
    
#     return token_dict

# print(train(sentence))

