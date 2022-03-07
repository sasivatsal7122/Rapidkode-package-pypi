import rapidkode as rd 

arr = [2, 3, 4, 10, 40]
x = 10


arr = [ 0, 1, 1, 2, 3, 5, 8, 13, 21,
    34, 55, 89, 144, 233, 377, 610 ]
x = 2333
tr = rd.ternary()
print(tr.search(arr,x,0,len(arr)-1))

