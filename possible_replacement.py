# load the query_auto_completion model and run it
class PossibleReplacement():
    def __init__(self, someVariable):
        self.someVariable = someVariable

    def someFunction(self):
        print(self.someVariable)


def possibleReplace():
    newObj = PossibleReplacement("PossibleReplacement")
    newObj.someFunction()
    print("yayyyyyy!")