# Create a class with a field self.my_var
# This class has a method called increment which increases the value of self.my_var by 1
# What should this method return if we want to execute my_object.increment().increment().increment() 
# and see that the value of my_object.my_var has increased by 3?

class MyClass:
    def __init__(self, my_var): 
        self.my_var = my_var
    
    def increment(self):
        self.my_var += 1
        return self

test_var= MyClass(1)
test_var.increment().increment().increment()
print(test_var.my_var)

# class MyClass:
#     count =0
#     def __init__(self, my_var): 
#         self.my_var = my_var
    
#     def increment(self): 
#         self.my_var = int(self.count +1)
#         return self.my_var*4

# variablee = MyClass(1)
# print(variablee.my_var)
# print(variablee.increment())