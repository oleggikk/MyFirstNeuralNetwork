num = int(input())
num_str = str(num)
count = 0
while (len(num_str)) > 1:
    for i in range(len(num_str)):
        count += num % 10
        num //= 10
    num = count
    num_str = str(num)
    count = 0
print(num)