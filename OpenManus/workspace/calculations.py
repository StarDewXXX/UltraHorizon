# Calculate sum from 3 to 666
sum_result = sum(range(3, 667))
print(f"(1) Sum of numbers from 3 to 666: {sum_result}")

# Calculate product from 4 to 10
from functools import reduce
import operator

product_result = reduce(operator.mul, range(4, 11))
print(f"(2) Product of numbers from 4 to 10: {product_result}")