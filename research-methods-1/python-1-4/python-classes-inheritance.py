## Create a class
class Student:
    '''A Student class.'''
    fee_rise_amt = 1.03
    def __init__(self, first, last, tuition_fee): 
        self.first = first 
        self.last = last
        self.email = first + "." + last + "@uni.com"
        self.tuition_fee = tuition_fee

    def full_name(self): 
        return "{} {}".format(self.first, self.last)
    def apply_fee_rise(self):
        self.tuition_fee = int(self.tuition_fee * self.fee_rise_amt)

## Create a subclass that inherits from the parent Student class
class Postgrad(Student):
    fee_rise_amt = 1.05 # apply a different rise to postgrad students
    
    def __init__(self, first, last, tuition_fee, mode): 
        super().__init__(first, last, tuition_fee) # parent class will handle those arguments
        self.mode = mode

# instantiate postgrads
postgrad_1 = Postgrad("Astrid", "Aaberg", 10000, "full-time")
postgrad_2 = Postgrad("Ruadh", "Fahy", 12000, "part-time")

# Apply tuition rise
# print(postgrad_1.tuition_fee)  # before rise
# postgrad_1.apply_fee_rise()
# print(postgrad_1.tuition_fee) # after rise

# print mode
print(postgrad_1.mode)


# print(help(Postgrad)) # to look up info about what is inherited etc.