try:
    f = open('badfile.txt')
    # manual exception
    if f.name == "badfile.txtx":
        raise Exception
except FileNotFoundError:
    print("Can't find the file!")
except Exception:
    print("Sorry, something went wrong.")
else:
    print(f.read())
    f.close()
finally:
    print("Executing finally")