def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print(args[0], args[1])
        print(result)
        print("After function call")
        return result
    return wrapper

@my_decorator
def my_function(x, y):
    return x + y


def main():
    print("hey there")
    print(my_function(1,2))


# Using the special variable 
# __name__
if __name__=="__main__":
    main()