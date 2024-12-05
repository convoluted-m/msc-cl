## PASSING DATA TO A FUNCTION

# PASS BY VALUE  
# When primitive data, e.g. bool, int, float, string or immutable data, e.g. tuple, 
# is passed to a function, the data is copied into the function.
# Thus, if the data is changed inside the function, it won't change outisde of the function. 
print(1, ['a', 'b'])

# PASS BY REFERENCE
# When mutable data type, e.g. list, set, or a dictionary are passed to a function, 
# the data isn't copied over - only a reference is passed to the function.
# Thus, if the data is changed inside the function, it changes outside as well.
number = 1
my_list = ['a', 'b']
print(number, my_list) # can access original data even after passing to the function


# Example
# Create some variables, define a function to change the variables a
# and see the effect depending on the data type.
def func(x, y):
    x = x -1
    y.pop()

if __name__ == "__main__": # this tells Python that everything after this is part of the main program
    i = 1 # primitive data type --> passed by value (i copied over to function and stored in x)
    l = ['a', 'b'] # mutable type --> passed by reference (change inside the function affects it)

func(i,l) # func(i,l.copy()) this won't affect the original list
print(i,l)  

# To avoid permanenet changes to mutable data types when passed to a function, 
# passs a copy of the data to the function  with .copy()

###
# Check if two variables reference the same data use is operator
my_list = ['a', 'b', 'c']
m = my_list
n = my_list.copy()
# check if they have the same values - Yes, both show True
m == my_list
n == my_list

# check if they reference the same data - No
my_list is m # True
n is my_list # False, they reference different types of data 

# modify the original list
my_list.pop()
print(my_list)

print(m) # this list was modified too because they reference the same data
print(n) # this was not since it's a copy of the original list

id(my_list)
id(m)
id(n)

###
# None indicates an absence of value
a = None
b = None
a==b # True - there is only one None
a is b #True

# None has its own data type
type(None) # NoneType
# We can use None with a variable that doesn't yet have a value