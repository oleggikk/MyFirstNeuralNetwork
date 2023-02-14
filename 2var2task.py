
def stringSearch(n: int, s: list) -> list:
    if (k == 1):
        return n
    count = 0
    a = []
    for i in range(0, n):
        b = []
        b.append(i)
        while (s[i] != 0):
            count += 1
            i += 1
            if (i == n):
                break
        b.append(i)
        b.append(count)
        a.append(b)
        count = 0
    return a

def stringFix(n: int, k: int, s: list) -> list:
    for i in range(0, n):
        if (s.count(s[i]) < k):
            s[i] = 0
    return s

def findMax(a: list, k: int) -> int:
    a.sort(key= lambda a: a[2], reverse=True)
    r = len(a)
    for i in range(0, r):
        if (stringFix(a[i][2], k, s[a[i][0]:a[i][1]]).count(0) == 0):
            return a[i][2]
        else:
            r += 1
            a[i][2] = 0
            a = a + stringSearch(a[i][2], a[i])
            a.sort(key=lambda a: a[2], reverse=True)

n, k = map(int, input().split())
s = list(input())
s = stringFix(n, k, s)
a = stringSearch(n, s)
print(findMax(a, k))


