######## TASK
# Investigate linguistic variation in the non-finite complementation after the verb “help”. Task: Collect the data from the BNC and Spoken BNC corpora. Create a dataset of KWIC (key word in context) in the form of concordance lines containing the word "help" in context. Search for 120 characters before and 240 characters after the "help" keyword (that is 15 raw words before help, 30 raw words after help). The resulting data should be saved into a csv file containing all "help" KWIC concordance lines from the corpora and should have the following columns: "Hit"as the very first column. It should give every KWIC example a unique number.
# 1) Find all instances of the root "help"(including nouns, adjectives like helpless, the verb help without an infinitive). 
# 2) Label the dependent variable ("help" keyword) as one of the following: "TO" (help to do), "BARE" (help do), "ING" (help + –ing), "INING" (help + in + - ing). Every example you extract must be coded for one of these four complementation patterns. If there is no complement clause or help is not a verb, label as "NA". consider 15 words to the right of help. Put this info in you dataset in a "DepVar" columns with the label.  
# 3) Extract independent variables. You should value every example for the following independent variables:Inflectional form of help,  Verb lemma of the non-finite complement clause, The number of words intervening between help and the head of its non-finite complement, Four additional columns for the object of help, Voice of help (passive or active), Preceding to (horror aequi), polarity (is help positive or negative), Two additional columns for the subject of help
# 4) Extract text-level metadata, e.g. TextID, year of creation.

# Import packages
import csv
# Import the ElementTree module from xml.etree package to parse XML files 
import xml.etree.ElementTree as ET

#### PREPARE DATA
# Create a function to read an XML file and store its data (from a local corpus)
def read_xml_file(file_path):
    tree = ET.parse(file_path) # parse and return an ET object
    root = tree.getroot() # retrieve the root element of the parsed tree
    return root

# Create a list of XML files to process 
xml_files = [
    '/Users/martynakosciukiewicz/Documents/source/msc-cl/corp-ling/BNC/Texts/A00.xml',
    '/Users/martynakosciukiewicz/Documents/source/msc-cl/corp-ling/BNC/Texts/A01.xml'
]  

##### 1. Find all instances of "help"
# Create a list to store all instances of "help" 
help_instances = []

# Iterate through all the files, find all words 
for file in xml_files:
    print(f"Reading file: {file}")
    tree = ET.parse(file)
    root = tree.getroot()
    print(f"Root tag: {root.tag}")

    words = tree.findall('.//w')
    print(f"Number of words found in {file}: {len(words)}")

    for word in words:
      # if word starts with "help", store it and its POS tag
      if word.text.lower().startswith('help'):
        actual_word = word.text
        pos_tag = word.get('c5') 
        help_instances.append((file, actual_word, pos_tag))
    print(f"Number of 'help' instances found in {file}: {len(help_instances)}")


### Create KWIC concordance lines and extract metadata
help_KWIC = [] # 120 characters before, 240 characters after "help" are required
help_form = [] # store the form of the hit, "help", "helped", "helping" etc.
help_pos = [] # store the Part of speech of each "help" hit
max_buffer = 10  # for 60 characters left of help, 10 words might typically be found

# Store file name
file_id = []
# Store the genre (from stext or wtext)
genre = []
# Store mode: "written" or "spoken" (from classCode)
mode = []
# Store subgenre (from classCode)
subgenre = []
# Store the year of text creation
year = []
# Store the index of each concordance line
hits = []
hit = 1 # for the first hit, then increment for each additional hit
corpus = [] # BNC
variety = [] # BrE

# Loop through XML files, parse with ElementTree and extract medatada
for file in xml_files:
   print(f"Now turning file {file} into an XML structure")
   tree = ET.parse(file)
   root = tree.getroot()

   id = file.replace('.xml', '') # file name

   wstext_element = root.find('.//wtext')
   if wstext_element is None:
    wstext_element = root.find('.//stext')
   genrevalue = wstext_element.get('type')

   classcode = root.find('.//classCode')
   meta = classcode.text

   creation = root.find('.//creation')
   yearvalue = creation.get('date')

   left_context = [] # 10 previous words
   collecting_right = False   # unless we find "help", we don't collect right context
   right_context = []
   current_right_context_words = 0

   # Iteratre through every word
   for elem in root.iter():
      if collecting_right: 
          if elem.tag in ['w', 'c']:
              right_context.append(elem.text)
              if elem.tag == 'w':  # only count words, not punctuation
                  current_right_context_words += 1
                  if current_right_context_words >= 20:  # stop after ~20 words
                      collecting_right = False  # stop collecting
                      right_KWIC = ' '.join(right_context)  
                      kwic_line = f"{left_KWIC} {help_word} {right_KWIC}"
                      help_KWIC.append(kwic_line)

      # If the element is a word or punctuation, add it to the left context
      if elem.tag in ['w', 'c'] and not elem.text.lower().startswith('help'):
        left_context.append(elem.text)
        # Keep buffer at max size
        if len(left_context) > max_buffer:
            left_context.pop(0)

      # Check if current elem is "help"; if yes, collect left and right context around each instance
      if elem.tag == 'w' and elem.text.lower().startswith('help'):
        left_KWIC = ' '.join(left_context) # join turns a list into a string and places ' ' in between each element
        # Get the text of this instance of "help", store it in help_form
        help_word = elem.text
        help_form.append(help_word)
        help_c5 = elem.get('c5') # get pos tag
        help_pos.append(help_c5)

        # add an instance to all the meta-information
        file_id.append(id)
        genre.append(genrevalue)

        if meta[0] == "W": # meta[0] will contain the first character, W or S
          mode.append("Written")
        if meta[0] == "S":
          mode.append("Spoken")
        subgenre.append(meta[2:])# everything after W/S and whitespace, add to genre
        year.append(yearvalue)
        corpus.append("BNC")
        variety.append("BrE")
        hits.append(str(hit)) # add the hit (as a string) to the list hits
        hit += 1 

        # collect the right context 
        collecting_right = True 
        right_context = []  # Reset right context
        current_right_context_words = 0  # Reset word counter

# # Debugging: Print the lengths of the lists
# print(f"hits: {len(hits)}, help_KWIC: {len(help_KWIC)}, help_form: {len(help_form)}, help_pos: {len(help_pos)}, file_id: {len(file_id)}, year: {len(year)}, variety: {len(variety)}, genre: {len(genre)}, subgenre: {len(subgenre)}, mode: {len(mode)}, corpus: {len(corpus)}")

# # Debugging: Print the first few elements of each list
# print("First few elements of each list:")
# print("hits:", hits[:5])
# print("help_KWIC:", help_KWIC[:5])
# print("help_form:", help_form[:5])
# print("help_pos:", help_pos[:5])
# print("file_id:", file_id[:5])
# print("year:", year[:5])
# print("variety:", variety[:5])
# print("genre:", genre[:5])
# print("subgenre:", subgenre[:5])
# print("mode:", mode[:5])
# print("corpus:", corpus[:5])


##delete
# # print the concordance
# output = ""
# for h, kwic, f, pos, id, date, var, gen, subgen, mod, corp in zip(hits, help_KWIC, help_form, help_pos, file_id, year, variety, genre, subgenre, mode, corpus):
#     output += f'{h}\t{kwic}\t{f}\t{pos}\t{id}\t{date}\t{var}\t{gen}\t{subgen}\t{mod}\t{corp}\n'

# # # Debugging - print output
# # print("First few lines of the output:")
# # print("\n".join(output.split("\n")[:10]))

# # Save the concordance
# with open("BNC_partial_results.txt", "w") as f:
#     f.write(output)

# # Debugging: Confirm the file was written
# print("Concordance saved to BNC_partial_results.txt")

with open("BNC_partial_results.csv", "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(['Hit', 'KWIC', 'Form', 'POS', 'FileID', 'Year', 'Variety', 'Genre', 'Subgenre', 'Mode', 'Corpus'])
    for h, kwic, f, pos, id, date, var, gen, subgen, mod, corp in zip(hits, help_KWIC, help_form, help_pos, file_id, year, variety, genre, subgenre, mode, corpus):
        csvwriter.writerow([h, kwic, f, pos, id, date, var, gen, subgen, mod, corp])