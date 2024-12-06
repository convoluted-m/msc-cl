# Create a class
class Student:
    '''A Student class.'''

    # Define a class variable (different than instance variable)
    num_students = 0  
    fee_rise_amt = 1.03

    def __init__(self, first, last, tuition_fee): 
        # Define instance variables
        self.first = first 
        self.last = last
        self.email = first + "." + last + "@uni.com"
        self.tuition_fee = tuition_fee
        Student.num_students += 1  # each time we create a new student, increment by 1

    # Define methods - regular method that takes the 'self' instance as an argument
    def full_name(self): 
        return "{} {}".format(self.first, self.last)
    
    def apply_fee_rise(self):
        self.tuition_fee = int(self.tuition_fee * self.fee_rise_amt)

    # Define a class method that takes class as an argument - uses @ decorator
    @classmethod
    def set_fee_rise_amt(cls, amount):
        cls.fee_rise_amt = amount

# Change fee_raise_amt from initial 1.03 - ovverides instance methods since it calls a class method
Student.set_fee_rise_amt(1.05)

# Create student instances
student_1 = Student("Astrid", "Aaberg", 10000)
student_2 = Student("Ruadh", "Fahy", 12000)

print(Student.num_students)
print(student_1.fee_rise_amt)
print(student_2.fee_rise_amt)
print(Student.fee_rise_amt)