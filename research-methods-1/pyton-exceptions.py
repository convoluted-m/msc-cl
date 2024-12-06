# Exceptions
# Antitipating which sections of code might throw an error at the user and handling those 
try:
    f = open('testfile.txt')
except FileNotFoundError:
    print("Can't find the file!")
except Exception:
    print("Sorry, something went wrong.")
else:
    print(f.read())
    f.close()
finally: # runs whether the code in try-except is successful or not
    print("Executing finally")