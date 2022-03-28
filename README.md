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

- You can replace with sys with ['bin','dec','oct','hex'] and new_sys with ['bin','dec','oct','hex'], to make number conversions.
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

rk.binary.show()
rk.binary.
```





                                                                                        
