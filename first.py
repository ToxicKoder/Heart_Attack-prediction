import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print("The Transpose of the array is\n",arr.T)
# print("The 2-D array is\n",arr)

# x = np.linspace(1 , 10 , 100)
# y1 = np.log(x)
# y2 = np.exp(x)

# plt.plot(x, y1)
# #plt.plot(x, y2)
# plt.title("exponential")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# plt.grid(True)
# plt.show()


# bar chart
# categories = ['A', 'B', 'C']
# values = [10, 20, 15]

# plt.bar(categories, values)
# plt.title("Bar Chart")
# plt.show()

# x = np.random.rand(50)
# y = np.random.rand(50)

# plt.scatter(x, y, color='red')
# plt.title("Scatter Plot")
# plt.show()

# s = pd.Series([10, 20, 30])
# print(s)

# data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
# df = pd.DataFrame(data)
# print(df)


# data = {'Name':['Nathan','Shubham','Jyoti'] , 'Age' : [12,34,45]}
# df = pd.DataFrame(data)
# print("The required data is \n",df)

df=pd.read_csv('Medicaldataset.csv')
print(df.info)
print(df.head(5))
print("The column names are",df.columns)
