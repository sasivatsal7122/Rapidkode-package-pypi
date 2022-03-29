<h1 align='center'>RapidKode: get it right the first time</h1>

#### RapidKode is a Python package that provides fast, flexible, and expressive data structures,alogorithms designed to make working with both competitive programming and coding  easy and intuitive. It aims to be the fundamental high-level building block for competitive programming in Python.With RapidKode you can perform complex algorithms in less time, all the algorithms are optimized to their finest to reduce time complexity to make you stand out in the leader board. There is no more time wasting writing huge chucks of code and debugging them later, With RapidKode everything happens at your fingertips with just a single line of code. The aim of Rapidkode is to help beginners get started in competative programing, understand the importance of time and space. The motto of making Rapidkode is to 'Get it right the first time' instead of spending 10's of precious minutes on util functions.

# Installation:

- ### Install from the official pypi website -> <a href=''>Click Here</a>
or 
```python
$ pip install rapidkode
```
### For issues,bug reports and contributions visit the development repo --> <a href='https://github.com/sasivatsal7122/Rapidkode-package-pypi'>Click Here</a>

# Available Functions :

## Number functions :
| **syntax**                     | **operation**                                            
|:------------------------------:|:--------------------------------------------------------:
| numbers.gen_sparsenum_upto(x)  | generates sparse number upto the given range             
| numbers.get_sparsenum_after(n)  | Returns the succeding sparse number for the given number
| numbers.checkprime(x)           | Returns True if number is prime                         
| numbers.getprimes.generate(x)  | Returns first x prime numbers                           
| numbers.getprimes.upto(x)      | Returns prime numbers upto given range                   
| numbers.getprimes.inrange(x,y) | Returns prime numbers in the given range                
| numbers.fib.getelement(x)      | Returns x'th fibonacci number                            
| numbers.fib.generate(x)         | Returns first x fibonacci numbers           

## Example:
```python
import rapidkode as rk

var = rk.numbers.gen_sparsenum_upto(100)
print(var)	

var = rk.numbers.get_sparsenum_after(3289)		
print(var)	

var = rk.numbers.checkprime(8364)	
print(var)	

var = rk.numbers.getprimes.generate(100)	
print(var)	

var = rk.numbers.getprimes.inrange(100,500)	
print(var)	

var = rk.numbers.fib.getelement(58)	
print(var)	

var = rk.numbers.fib.generate(25)	
print(var)
```
## Output:
```
[0, 1, 2, 4, 5, 8, 9, 10, 16, 17, 18, 20, 21, 32, 33, 34, 36, 37, 40, 41, 42, 64, 65, 66, 68, 69, 72, 73, 74, 80, 81, 82, 84, 85, 128]

4096

False

[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541]

[101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]

365435296162

[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368]

```

## Number System - Conversion functions:
| **Syntax**                     | **Operation**                          
|:------------------------------:|:---------------------------------------
| convert(x,'sys').to('new_sys') | Converts x from sys to new sys          
| Example:                       |                                         
| convert(9845,'dec').to('bin')  | Converts 9845 from Decimal to Binary    
| convert(3745,'oct').to('hex')  | Converts 3745 from Octal to Hexadecimal 

- You can replace with `sys` with ['bin','dec','oct','hex'] and `new_sys` with ['bin','dec','oct','hex'], to make number conversions.
```python
import rapidkode as rk

converted_num_1 = rk.convert(2013,'dec').to('bin')
print(converted_num_1)
converted_num_2 = rk.convert(11111011101,'bin').to('hex')
print(converted_num_2)
converted_num_3 = rk.convert('7dd','hex').to('dec')
print(converted_num_3)
converted_num_4 = rk.convert(5634,'oct').to('dec')
print(converted_num_4)
converted_num_5 = rk.convert(2972,'hex').to('oct')
print(converted_num_5)
converted_num_6 = rk.convert(24562,'oct').to('bin')
print(converted_num_6)
```
#### Output:
```
11111011101
7dd
2013
2972
24562
10100101110010
```
## Searching Algorithms :
| **Technique**        | **Syntax**                  | **Operation**                | **Time Complexity** |
|:--------------------:|:---------------------------:|:----------------------------:|:-------------------:|
| Linear Search        | linear.search(arr,x)        | Returns position of x in arr | O(n)                |
| Binary Search        | binary.search(arr,x)        | Returns position of x in arr | O(log n)            |
| Jump Search          | jump.search(arr,x)          | Returns position of x in arr | O(√ n)              |
| Interpolation Search | interpolation.search(arr,x) | Returns position of x in arr | O(log2(log2 n))     |
| Exponential Search   | exponential.search(arr,x)   | Returns position of x in arr | O(log2 i)           |
| Ternary Search       | ternary.search(arr,x)       | Returns position of x in arr | O(log3 n)           |

additonally you can use:
| Function | Operation                       
|----------|---------------------------------
| .show()  | prints the code in the terminal 
| .info()  | Gives the brief info            
| .algo()  | prints the step wise algorithm  

#### Example:
```python
import rapidkode as rk

>>> rk.binary.show()
>>> rk.binary.info()
>>> rk.binary.algo()
```
#### Output:
```python
def binarysearch(arr, x):
	l = 0
	r = len(arr)-1
	while l <= r:

		mid = l + (r - l) // 2
		if arr[mid] == x:
			return mid
		elif arr[mid] < x:
			l = mid + 1
		else:
			r = mid - 1
	return "element not found"
```
```
Binary search is the search technique that works efficiently on sorted lists
. Hence, to search an element into some list using the binary search technique, we must ensure that the list is sorted
. Binary search follows the divide and conquer approach in which the list is divided into two halves, and the item is compared with the middle element of the list
. If the match is found then, the location of the middle element is returned
. Otherwise, we search into either of the halves depending upon the result produced through the match
```
```
Algorithm

Step 1 - Read the search element from the user.

Step 2 - Find the middle element in the sorted list.

Step 3 - Compare the search element with the middle element in the sorted list.

Step 4 - If both are matched, then display "Given element is found!!!" and terminate the function.

Step 5 - If both are not matched, then check whether the search element is smaller or larger than the middle element.

Step 6 - If the search element is smaller than middle element, repeat steps 2, 3, 4 and 5 for the left sublist of the middle element.

Step 7 - If the search element is larger than middle element, repeat steps 2, 3, 4 and 5 for the right sublist of the middle element.

Step 8 - Repeat the same process until we find the search element in the list or until sublist contains only one element.

Step 9 - If that element also doesn't match with the search element, then display "Element is not found in the list!!!" and terminate the function.
```
The three functions `.show()`, `.info()`, `.algo()` can be used for all the 6 searching techniques.

## Sorting Algorithms :

| **Technique**   | **Syntax**                | **Operation**                     | **Time Complexity** |
|:---------------:|:-------------------------:|:---------------------------------:|:-------------------:|
| Selection Sort  | selection.sort(arr)       | Sorts and Returns the given array | O(n^2)              |
| Bubble Sort     | bubble.sort(arr)          | Sorts and Returns the given array | O(n^2)              |
| Insertion Sort  | insertion.sort(arr)       | Sorts and Returns the given array | O(n^2)              |
| Merge Sort      | merge.sort(arr)           | Sorts and Returns the given array | O(n log(n))	        |
| Heap Sort       | heap.sort(arr)            | Sorts and Returns the given array | O(n log(n))         |
| Quick Sort      | quick.sort(start,end,arr) | Sorts and Returns the given array | O(n^2)              |
| Count Sort      | count.sort(arr)           | Sorts and Returns the given array | O(n+k)              |
| Radix Sort      | radix.sort(arr)           | Sorts and Returns the given array | O(nk)               |
| Bucket Sort     | bucket.sort(arr)          | Sorts and Returns the given array |  O(n + k)           |
| Shell Sort      | shell.sort(arr)           | Sorts and Returns the given array | O(nlog n)           |
| Comb Sort       | comb.sort(arr)            | Sorts and Returns the given array | O(n log n)          |
| Pigeongole Sort | pigeonhole.sort(arr)      | Sorts and Returns the given array |  O(n + N)           |
| Cycle Sort      | cycle.sort(arr)           | Sorts and Returns the given array | O(n2)               |

additonally you can use:
| Function | Operation                       
|----------|---------------------------------
| .show()  | prints the code in the terminal 
| .info()  | Gives the brief info            
| .algo()  | prints the step wise algorithm  

#### Example:
```python
import rapidkode as rk

>>> rk.count.show()
>>> rk.count.info()
>>> rk.count.algo()
```
#### Output:
```python
def countsort(arr):
    output = [0 for i in range(len(arr))]
    count = [0 for i in range(256)]
    array = [0 for _ in arr]
    for i in arr:
        count[i] += 1
    for i in range(256):
        count[i] += count[i-1]
    for i in range(len(arr)):
        output[count[arr[i]]-1] = arr[i]
        count[arr[i]] -= 1
    for i in range(len(arr)):
        array[i] = output[i]
    return array
```
```
Counting sort is a sorting algorithm that sorts the elements of an array by counting the number of occurrences of each unique element in the array
. The count is stored in an auxiliary array and the sorting is done by mapping the count as an index of the auxiliary array
. Counting sort is a sorting technique based on keys between a specific range
. It works by counting the number of objects having distinct key values (kind of hashing)
. Then doing some arithmetic to calculate the position of each object in the output sequence
```
```
Algorithm

step 1 - Find out the maximum element (let it be max) from the given array.

step 2 - Initialize an array of length max+1 with all elements 0.
         This array is used for storing the count of the elements in the array.

step 3 - Store the count of each element at their respective index in count array

         For example: if the count of element 3 is 2 then, 2 is stored in the 3rd position of count array.
         If element "5" is not present in the array, then 0 is stored in 5th position.

step 4 - Store cumulative sum of the elements of the count array.
         It helps in placing the elements into the correct index of the sorted array.

step 5 - Find the index of each element of the original array in the count array.
         This gives the cumulative count. Place the element at the index calculated as shown in figure below.

step 6 - After placing each element at its correct position, decrease its count by one.

```
The three functions `.show()`, `.info()`, `.algo()` can be used for all the 13 sorting techniques.

## Graph Functions :
| **Syntax**          | **Operation**                               
|:-------------------:|:-------------------------------------------:
| .buildedge(u,v)     | Create an Graph Edge                        
| .buildmultiedge([]) | Creates an Graph with given Coord list      
| .BFS(x)             | Perform Breadth First Search               
| .DFS(x)             | Performs Depth First Search                 
| .findAP()           | Returns the Articulation Point of the graph

### Example:
```python
import rapidkode as rk

# make graph object with graph() class

my_graph = rk.graph()

# adding one edge at a time
my_graph.buildedge(0, 1)
my_graph.buildedge(0, 2)
my_graph.buildedge(1, 2)
my_graph.buildedge(2, 0)
my_graph.buildedge(2, 3)
my_graph.buildedge(3, 3)
```

```python
import rapidkode as rk

# make graph object with graph() class

my_graph = rk.graph()

# adding multiple edges at once

my_graph.buildmultiedge([0,1,0,2,1,2,2,0,2,3,3,3])
```
```python
import rapidkode as rk

# make graph object with graph() class

my_graph = rk.graph()

my_graph.buildmultiedge([0,1,0,2,1,2,2,0,2,3,3,3])

# performing BFS from edge 2
print(my_graph.BFS(2))

# performing DFS from edge 2
print(my_graph.DFS(2))

# finding the Articulation Point
print(my_graph.findAP())

```
### Output:
```
['-->', 2, '-->', 0, '-->', 3, '-->', 1]

['-->', 2, '-->', 0, '-->', 1, '-->', 3]

2
```
## Pattern Functions :

The following function uses Rabin-Karp algorithm which is an algorithm used for searching/matching patterns in the text using a hash function. Unlike Naive string matching algorithm, it does not travel through every character in the initial phase rather it filters the characters that do not match and then performs the comparison.

| Syntax                    | Operation                                          |
|---------------------------|----------------------------------------------------|
| pattern.isthere(a).inn(b) | Returns true if string a is present in string b    |
| pattern.whereis(a).inn(b) | Returns the index position of string a in string b |

### Example:
```python
import rapidkode as rk

a = 'sasi'
b = 'satyasasivatsal'

print(rk.isthere(a).inn(b))
print(rk.whereis(a).inn(b))
```
### Output:
```python
True

4
```
## Linkedlist Functions:

| **Operation**             | **Syntax**                                       |
|-----------------------|----------------------------------------------|
| .ins_beg(node)        | Inserts a new node at beginning              |
| .ins_end(node)        | Inserts a new node at the end                |
| .ins_after(pos,node)  | Inserts a new node after the node specified  |
| .ins_before(pos,node) | Inserts a new node before the specified node |
| .del_node(node) | Deletes the specified node |
| .return_as_list() | Returns LinkedList as python List |

### Example:
```python
import rapidkode as rk

my_list = rk.linkedlist()

my_list.head = rk.node('a')

s1 = rk.node('b')
s2 = rk.node('c')
s3 = rk.node('d')
s4 = rk.node('e')
s5 = rk.node('f')
s6 = rk.node('g')

my_list.head.next = s1
s1.next =  s2
s2.next =  s3
s3.next =  s4
s4.next =  s5
s5.next =  s6

print(my_list)
```
### Output :
```
a -> b -> c -> d -> e -> f -> g -> None
```
### Example -2 :
```python
# insertion at beginning
my_list.ins_beg(rk.node('A'))

# insertion at end
my_list.ins_end(rk.node('G'))

# insertion at positiom
my_list.ins_after('e',rk.node('E'))

# insertion at position
my_list.ins_before('c',rk.node('C'))

# deletion of ndoe
my_list.del_node('b')

# returning as list
my_listt = my_list.return_as_list()

print(my_list)

print(my_listt)
```
### Output :
```
A -> a -> C -> c -> d -> e -> E -> f -> g -> G -> None

['A', 'a', 'C', 'c', 'd', 'e', 'E', 'f', 'g', 'G', 'None']
```
                                                                                        
## Bit manipulation Fuctions:

| **Syntax**               | **Operation**                              |
|--------------------------|--------------------------------------------|
| bits.toggle_bits(x)      | Toggles the set bits and non set bits      |
| bits.convert_to_bin(x)   | Converts a given number into binary        |
| bits.counsetbits(x)      | Returns the no.of set bits in a dec number |
| bits.rotate_byleft(x,d)  | Rotates the bits to left by d times        |
| bits.rotate_byright(x,d) | Rotates the bits to left by d times        |
| bits.countflips(x,y)     | Returns the no.of flips to make x as y     |

### Example:
```python
import rapidkode as rk

var = rk.bits.toggle_bits(873652)
print(var)

var = rk.bits.convert_to_bin(873652)
print(var)

var = rk.bits.countsetbits(873652)
print(var)

var = rk.bits.rotate_byleft(873652,4)
print(var)

var = rk.bits.rotate_byright(873652,4)
print(var)

var = rk.bits.countflips(8934756,873652)
print(var)
```

### Output:
```
960632

11010101010010110100

8474306

13978432

54603

7
```

## Other misc Functions:
| **Syntax**                   | **Operation**                                           |
|------------------------------|---------------------------------------------------------|
| .showsieves()                | Prints Sieves code for finding prime number in terminal |
| getprimefactors.fornum(x)    | Returns a list of prime factors for given number        |
| findgcdof(x,y)               | Returns GCD of the given numbers                        |
| findinversions.forr(arr)     | Returns how close the array is from being sorted        |
| catlan_numbers.getelement(x) | Returns the x'th Catlan Number                          |
| catlan_numbers.gen(x)        | Returns a list of first x Catlan_numbers                |

### Example:
```python
import rapidkode as rk

var = rk.getprimefactors.fornum(6754)
print(var)

var = rk.findgcdof(97345435,8764897)
print(var)

var = rk.findinversions.forr([1, 20, 6, 4, 5])
print(var)

var = rk.catlan_numbers.getelement(15)
print(var)

var = rk.catlan_numbers.gen(28)
print(var)
```

### Output:
```
[2, 11, 307.0]

1

5

9694845.0

[1.0, 1.0, 2.0, 5.0, 14.0, 42.0, 132.0, 429.0, 1430.0, 4862.0, 16796.0, 58786.0, 208012.0, 742900.0, 2674440.0, 9694845.0, 35357670.0, 129644790.0, 477638700.0, 1767263190.0, 6564120420.0, 24466267020.0, 91482563640.0, 343059613650.0, 1289904147324.0, 4861946401452.0, 18367353072152.0, 69533550916004.0]
```

### Contributing to RapidKode
- All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
- Raise an issue if you find any problem
- If you like to contribute, fork the repo raise an issue before making a pull request, it will be easy for managing
- Logo and header credits --> <a href='https://github.com/HarshaMalla'>M.Sri Harsha</a>❤️


## Happy Rapid Koding!! 
