# GENERATORS yield one result at a time, but doesn't store them in memory.
# Hence, they are better at performance than, e.g. lists. That matters when dealing with loads of values.
# unlike return, you don’t exit the function afterward - the state of the function is remembered. 
# when next() is called, the previously yielded variable num is incremented and  yielded again

# ## The standard way with a return statement
# def sqaure_numbers(nums):
#     result=[]
#     for i in nums:
#         result.append(i*i)
#     return result

# my_nums = sqaure_numbers([1,2,3,4,5])
# print(my_nums)

## converted into a generator
def sqaure_numbers(nums):
    for i in nums:
        yield (i*i) # yield keywords makes it a generator

my_nums = sqaure_numbers([1,2,3,4,5])
print(my_nums) # returns a generator object, doesn't yield results yet; "generator object at..." message

# We can print the values with the next()method 
# print(next(my_nums)) 
# print(next(my_nums))
# print(next(my_nums))
# print(next(my_nums))
# print(next(my_nums))

# or with a for loop
for num in my_nums:
    print(num)

# ## we could also use the function using a list comprehension
# my_nums = [x*x for x in [1,2,3,4,5]]
# # and use the for loop
# for num in my_nums:
#     print(num)

## list comprehension turned into a generator - generator expression
my_nums = (x*x for x in [1,2,3,4,5])
print(my_nums) # gives "generator object at..." message
# # so we need to run the for loop to get results - they are not stored in memory though
# for num in my_nums:
#     print(num)

## instead of the for loop could put the results in a list to store them
my_nums = (x*x for x in [1,2,3,4,5])
print(list(my_nums)) # now the results are stored in a list