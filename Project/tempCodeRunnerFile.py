letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
n = int(input())
length = 1
while (n - 1) // 26 ** length:
    n -= 26 ** length
    length += 1
rez = ''
for i in range(length):
    ind = (n - 1) // 26 ** (length - i - 1)
    rez += letters[ind]
    n -= 26 ** (length - i - 1) * ind

print(rez)