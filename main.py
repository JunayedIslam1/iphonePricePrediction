import pandas
from sklearn.linear_model import LinearRegression
data = pandas.read_csv('iphone_price.csv')
model = LinearRegression()
model.fit(data[['version']], data[['price']])
print(model.predict([[20]]))
print(model.predict([[24]]))


# import pandas
# import matplotlib.pyplot as plt
# data = pandas.read_csv('iphone_price.csv')
# # plt.scatter(data['version'], data['price'])
# plt.bar(data['version'], data['price'])
# plt.show()
