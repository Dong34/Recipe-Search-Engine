# load the query_auto_completion model and run it
class QueryAutoCompletion():
    def __init__(self, someVariable):
        self.someVariable = someVariable

    def someFunction(self):
        print(self.someVariable)


def main():
    newObj = QueryAutoCompletion("SI650")
    newObj.someFunction()
    print("yayyyyyy!")


if __name__=="__main__":
    main()