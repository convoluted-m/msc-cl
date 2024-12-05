# ### DATA STRUCTURES

# ## LISTS
# # mutable, can mix data types
# # indexes, lists within lists
# m = [3,4,5]
# n = [6,7,8]
# l=[m,n]
# print(l)
# print(l[0][1])

# # append to the end of the list
# a = [1,2,3]
# print(a)
# a.append(4)
# print(a)
# # extend for adding several values
# a.extend([5,6])
# print(a)

# #remove elements: remove() amd pop()
# a.remove(6)
# print(a)

# a.pop(0)
# print(a)

# # Assignment vs copying
# l = ['a', 'b']
# m = l # l and m point to the same list
# n = l.copy() # n is a new list that has copied the elements from l
# print(l,m,n)
# # a change to the original list l dpoesn't affect the copied list
# l[0] = 'z'
# print(l,m,n)

# # TUPLES
# # immutable, can mix data dypes
# t = (1,2,3,4,5,6,7,8,9)
# print(t[0:2]) # slicing to get subtuples
# # we can add tuples
# t2 = (10, 11)
# print(t + t2)
# # tuples can store lists
# list1 = ['a', 'b']
# list2 = ['c', 'd']
# tuple_of_lists = (list1, list2)
# print(tuple_of_lists)
# # access first list in the tuple
# print(tuple_of_lists[0])
# # access the first element in the first list
# print(tuple_of_lists[0][0])

# # tuple with one element
# one_tuple = (1,)

# # turn a lit into a tuple with tuple()
# l = ['a', 'b']
# list_to_tuple = tuple(l)
# print(list_to_tuple)

# # list to tuple
# tuple_to_list = list(list_to_tuple)
# print(tuple_to_list)

# # check if an element is in a tuple
# tuple_to_check = ('hey', 'how', 'are', 'you')
# print('hey' in tuple_to_check)

# # SETS
## mutable, unordered, unique elements
# s = {1,2,3}
## we can't use indices for accessing elements but can use the 'in' operator
# print(1 in s)
# add elements
# s.add(4)
# print(s)
## add a set to a set
# s2 = {'daisy', 'tree'}
# s.update(s2)

## add elements from a lit to a set
# list_to_add = [5,6,7]
# s.update(list_to_add)
# print(s)

## add elements from a tuple to a set
# t = ('a')
# s.update(t)
# print(s)
## we can't allow duplicates in sets

## remove elements
# s.pop()
# print(s)

## turn a list/ tuple into  a set
# l=['s', 'o']
# set(l)
# print(l)

# t=('s', 'o')
# set(t)
# print(t)

# Combining sets
# union - combine all elements from sets
# a = {1,2,3}
# b = {4,5,6}
# print(a.union(b))

# c = {7,8,9}
# d = {10,11}
# print(c | d)

# # # Intersection - only elements that are on both sets
# s = {'a', 'b'}
# m = {'a', 'c'}
# # print(s.intersection(m))

# # # difference
# # print(s.difference(m))

# # symmetric difference is the union of sets
# print(s.symmetric_difference(m))

# ### DICTIONARIES
# d = {"name": "Capo", "age": "5"}
# # access elements using the key
# print(d["name"])

# # zip two lists to create a dict
# keys = ["name", "age"]
# values = ["Gnocchi", "5"]
# d = dict(zip(keys, values))
# print(d)

# # check values
# my_dict = {'books': 2, 'teas': 4}
# print(my_dict.values())
# # check if something is in the values with "in" operator
# print(2 in my_dict.values())

# # counter with a for loop
# word = "Gnoc Gnoc"
# counter= {}

# for letter in word:
#     if letter not in counter:
#         counter[letter] = 0
#     counter[letter] += 1

# print(counter)

# counter with defaultdict
from collections import defaultdict
my_word = "fossil"
counter = defaultdict(int)

for letter in my_word:
    counter[letter] += 1

print(counter)
