<h1 align='center'>RapidKode: get ir right the first time</h1>

##### RapidKode is a Python package that provides fast, flexible, and expressive data structures,alogorithms designed to make working with both competitive programming and coding  easy and intuitive. It aims to be the fundamental high-level building block for competitive programming in Python.With RapidKode you can perform complex algorithms in less time, all the algorithms are optimized to their finest to reduce time complexity to make you stand out in the leader board. There is no more time wasting writing huge chucks of code and debugging them later, With RapidKode everything happens at your fingertips with just a single line of code. The aim of Rapidkode is to help beginners get started in competative programing, understand the importance of time and space. The motto of making Rapidkode is to 'Get it right the first time' instead of spending 10's of precious minutes on util functions.

# Available Functions :

## Number functions :
| **syntax**                     | **operation**                                            
|:------------------------------:|:--------------------------------------------------------:
| numbers.gen_sparsenum_upto(x)  | generates sparse number upto the given range             
| number.gen_sparsenum_upto(x)   | Returns the succeding sparse number for the given number
| number.checkprime(x)           | Returns True if number is prime                         
| numbers.getprimes.generate(x)  | Returns first x prime numbers                           
| numbers.getprimes.upto(x)      | Returns prime numbers upto given range                   
| numbers.getprimes.inrange(x,y) | Returns prime numbers in the given range                
| numbers.fib.getelement(x)      | Returns x'th fibonacci number                            
| number.fib.generate(x)         | Returns first x fibonacci numbers           

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


                                                                                        
