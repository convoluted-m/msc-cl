# ## enumerate() returns a sequence (index and value) of tuples
# flowers = ['poppy', 'sunflower', 'daisy', 'aster']
# # print(list(enumerate(flowers)))

# # can be used in a for loop to return the object and index
# for index,flower in enumerate(flowers):
#     print(index, flower)

## count - plus one
flowers = ['poppy', 'sunflower', 'daisy', 'aster']
count = 1

for flower in flowers:
    print(count, flower)
    count +=1

# ## enumerate to store both index and value in, e.g. a dictionary
# flowers = ['poppy', 'sunflower', 'daisy', 'aster']
# flower_dict  = {flower:[] for flower in set(flowers)}
# print(flower_dict)
# # enumerate will allow to count the position of the values
# for index, flower in enumerate(flowers):
#     flower_dict[flower].append(index)
# print(flower_dict)