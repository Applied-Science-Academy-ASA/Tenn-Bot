import os

def pwd(file):
    #returning parent folder name of a defined file path
    return os.path.dirname(os.path.realpath(file))

if __name__ == "__main__":
    print(pwd(__file__))
    print(pwd(pwd(__file__)))
