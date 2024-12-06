# Create a class
class Student:
    '''A Student class.'''
    # Intialise an instance of the class
    def __init__(self, first, last, tuition_fee): 
        # Define instance variables
        self.first = first 
        self.last = last
        self.email = first + "." + last + "@uni.com"
        self.tuition_fee = tuition_fee

    # Define methods - regular method that takes the 'self' instance as an argument
    def full_name(self): 
        return "{} {}".format(self.first, self.last)

    def apply_fee_rise(self):
        self.tuition_fee = int(self.tuition_fee * self.fee_rise_amt)

# Create student instances and print the info
student_1 = Student("Astrid", "Aaberg", 10000)
student_2 = Student("Ruadh", "Fahy", 12000)
print(student_1.email)
print(student_1.full_name())
print(student_2.email)
print(student_2.full_name())