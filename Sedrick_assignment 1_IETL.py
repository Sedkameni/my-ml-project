## 1 - Working with Stacks
# Create a stack that initially contains the elements [10, 20, 30, 40].
stack = [10, 20, 30, 40]
print("Stack after initializing:", stack)

# Add a new element 50 to the end of the stack.
stack.append(50)
print("Stack after pushing 50:", stack)

# Remove the element 20 from the stack.
stack.remove(20)
print("Stack after removing the element 20:", stack)


# Traverse the stack and print each element
print("Traversing the stack:")
for item in stack:
    print(item)

## 2 - Classifying Data Types: Classify different types of data as structured, semi-structured, or unstructured.
## - The dataset 1: A spreadsheet with columns for EmployeeID, Name, Position, and Salary.
# Is a structured data because it is a spreadsheet which is a data organized in a predictable format with a fixed schema.

## - The dataset 2: An XML document representing a book's details, including Title, Author, and ISBN.
## Is a semi-structured data because data is an XML file which is data that doesn't fit into a rigid structure but has some organizational properties.

## -  The dataset 3 : A collection of customer feedback text files.
## Is an unstructured data because this data is a text file which is a data without a predefined data model or organization.




## 3- Creating and Querying a Relational Database: Design a relational database and perform SQL queries.
## - Create an SQLite database with two tables: Products and Orders.
###### - Use SQLite to create a database named `store.db`.
import sqlite3

### Create a new SQLite database and a table
conn = sqlite3.connect('shop.db')
cursor = conn.cursor()

#### Create the Products table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Products (
    ProductID INTEGER PRIMARY KEY,
    ProductName TEXT NOT NULL,
    Price REAL NOT NULL,
    Stock INTEGER NOT NULL
)
''')

#### Create the Orders table
cursor.execute('''
CREATE TABLE IF NOT EXISTS Orders (
    OrderID INTEGER PRIMARY KEY,
    CustomerName Text,
    ProductID INTEGER NOT NULL,
    Quantity INTEGER NOT NULL,
    OrderDate TEXT NOT NULL,
    FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
)
''')
conn.commit()
print("Database 'shop.db' with tables 'Products' and 'Orders' has been created successfully!")


#### - Populate Products table with 4 products
products = [
    (1, 'Laptop', 1200.00, 10),
    (2, 'Smartphone', 800.00, 25),
    (3, 'Headphones', 150.00, 50),
    (4, 'Keyboard', 70.00, 30)
]

cursor.executemany('''
INSERT OR IGNORE INTO Products (ProductID, ProductName, Price, Stock)
VALUES (?, ?, ?, ?)
''', products)

#### - Populate Orders table with 3 orders
orders = [
    (1, 'Alice', 1, 1, '2025-09-02'),
    (2, 'Bob', 3, 2, '2025-09-03'),
    (3, 'Charlie', 4, 1, '2025-09-04')
]

cursor.executemany('''
INSERT OR IGNORE INTO Orders (OrderID, CustomerName, ProductID, Quantity, OrderDate)
VALUES (?, ?, ?, ?, ?)
''', orders)


# Commit changes and close the connection
conn.commit()

print("Products and Orders tables have been populated successfully!")

## - Retrieve all orders with the corresponding product names and total price (quantity * price)
query=''' SELECT 
    o.OrderID,
    o.CustomerName,
    p.ProductName,
    o.Quantity,
    p.Price,
    (o.Quantity * p.Price) AS TotalPrice,
    o.OrderDate
FROM Orders o
JOIN Products p ON o.ProductID = p.ProductID
'''
cursor.execute(query)
result = cursor.fetchall()
print("All orders with the Corresponding product names and total prices are: ")
print(result)

### - Find orders where the total price exceeds $100
query2 ='''SELECT 
    o.OrderID,
    o.CustomerName,
    p.ProductName,
    o.Quantity,
    p.Price,
    (o.Quantity * p.Price) AS TotalPrice,
    o.OrderDate
FROM Orders o
JOIN Products p ON o.ProductID = p.ProductID
WHERE (o.Quantity * p.Price) > 100
'''
cursor.execute(query2)
selecto = cursor.fetchall()
print("Orders where the total price exceeds $100 are:")
print(selecto)

# Close the connection
conn.close()

### 4 - Designing an Entity Relationship Diagram (ERD): Create an ERD based on a given scenario.

