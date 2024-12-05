## CLASSES & OBJECTS
# define a class
class Student:
    def __init__(self, first, last, id):
        self.first = first
        self.last = last
        self.id = id
        self.email = first + "." + last + "@uni.com"

    def full_name(self):
        return "{} {}".format(self.first, self.last)

# create an instance of a class 
student_1 = Student("Astrid", "Aaberg", 834572)

print(student_1.email)
print(student_1.full_name())

