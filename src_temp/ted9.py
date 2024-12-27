from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name: str
    age: int
    email: str

# Valid input
user_data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
user = User(**user_data)  # Validates and creates a User instance


print(user.name)
print(user.age)
print(user.email)
# Invalid input
invalid_data = {"name": 123, "age": "thirty", "email": "invalid_email"}
try:
    invalid_user = User(**invalid_data)
except ValidationError as e:
    print(e)  # Prints detailed error messages