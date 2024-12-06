# STRING FORMATTING
person  = {'name': 'Jen', 'age': 46}

# string concatenation - not great
# sentence = 'My name is ' + person['name']
# print(sentence)

# # format method
# sentence = "My name is {}".format(person['name'])
# print(sentence)

# # f-strings
# first_name = 'Aria'
# last_name = "Prima"
# sentence = f"My name is {first_name} {last_name}."
# print(sentence)

# pet_name = "capo"
# pet_name_sent = f"{pet_name.upper()} wants to go for a walk."
# print(pet_name_sent)

plant = {'name':'chamomile', 'number':5}
plant_sent = f"We have {plant['name']} {plant['number']} plants."
print(plant_sent)