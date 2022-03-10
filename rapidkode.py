import math
from heapq import heappop, heappush
import random
from collections import defaultdict
from bisect import bisect, insort

''' Searching algorithms '''


class linear:
    def search(arr, x):
        for i in range(0, len(arr)):
            if (arr[i] == x):
                return i
        return "element not found"

    def show():
        print("""def linearsearch(arr, n, x):
	for i in range(0, n):
		if (arr[i] == x):
			return i
	return "element not found" """)

    def info():
        print("""
Linear search is a very simple search algorithm
. In this type of search, a sequential search is made over all items one by one
. Every item is checked and if a match is found then that particular item is returned, otherwise the search continues till the end of the data collection


              """)

    def algo():
        print("""
Algorithm

Linear Search ( Array A, Value x)

Step 1: Set i to 1

Step 2: if i > n then go to step 7

Step 3: if A[i] = x then go to step 6

Step 4: Set i to i + 1

Step 5: Go to Step 2

Step 6: Print Element x Found at index i and go to step 8

Step 7: Print element not found

Step 8: Exit
		""")


class binary:

    def search(arr, x):
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

    def show():
        print("""def binarysearch(arr, x):
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
	return "element not found" """)

    def info():
        print("""
Binary search is the search technique that works efficiently on sorted lists
. Hence, to search an element into some list using the binary search technique, we must ensure that the list is sorted
. Binary search follows the divide and conquer approach in which the list is divided into two halves, and the item is compared with the middle element of the list
. If the match is found then, the location of the middle element is returned
. Otherwise, we search into either of the halves depending upon the result produced through the match


              """)

    def algo():
        print("""
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
		""")


class jump:

    def search(arr, x):
        n = len(arr)
        step = math.sqrt(n)
        prev = 0
        while arr[int(min(step, n)-1)] < x:
            prev = step
            step += math.sqrt(n)
            if prev >= n:
                return -1

        while arr[int(prev)] < x:
            prev += 1
            if prev == min(step, n):
                return -1
        if arr[int(prev)] == x:
            return prev

        return "element not found"

    def show():
        print("""
def jumpsearch(arr, x):
	n = len(arr)
	step = math.sqrt(n)
	prev = 0
	while arr[int(min(step, n)-1)] < x:
		prev = step
		step += math.sqrt(n)
		if prev >= n:
			return -1

	while arr[int(prev)] < x:
		prev += 1
		if prev == min(step, n):
			return -1
	if arr[int(prev)] == x:
		return prev

	return "element not found" """)

    def info():
        print("""
  Jump search technique also works for ordered lists
. It creates a block and tries to find the element in that block
. If the item is not in the block, it shifts the entire block
. The block size is based on the size of the list
. If the size of the list is n then block size will be √n
. After finding a correct block it finds the item using a linear search technique
. The jump search lies between linear search and binary search according to its performance""")

    def algo():
        print("""
Algorithm

Step 1 - if (x==A[0]), then success, else, if (x > A[0]), then jump to the next block.

Step 2 - if (x==A[m]), then success, else, if (x > A[m]), then jump to the next block.

Step 3 - if (x==A[2m]), then success, else, if (x > A[2m]), then jump to the next block.

Step 4 - At any point in time, if (x < A[km]), then a linear search is performed from index A[(k-1)m] to A[km]
		""")


class interpolation:

    def search(A, target):
        if not A:
            return "element not found"

        (left, right) = (0, len(A) - 1)

        while A[right] != A[left] and A[left] <= target <= A[right]:
            mid = left + (target - A[left]) * \
                (right - left) // (A[right] - A[left])
            if target == A[mid]:
                return mid

            elif target < A[mid]:
                right = mid - 1

            else:
                left = mid + 1

        if target == A[left]:
            return left

        return "element not found"

    def show():
        print("""
def interpolationsearch(self, arr, lo, hi, x):

	if (lo <= hi and x >= arr[lo] and x <= arr[hi]):
		pos = lo + ((hi - lo) // (arr[hi] - arr[lo]) * (x - arr[lo]))
		if arr[pos] == x:
			return pos
		if arr[pos] < x:
			return self.search(self, arr, pos + 1, hi, x)
		if arr[pos] > x:
			return self.search(self, arr, lo, pos - 1, x)
	return "element not found" """)

    def info():
        print("""
Interpolation search is an algorithm for searching for a key in an array that has been ordered by numerical values assigned to the keys (key values)
. It was first described by W.W Peterson in 1957
. Interpolation search resembles the method by which people search a telephone directory for a name (the key value by which the book's entries are ordered): in each step the algorithm calculates where in the remaining search space the sought item might be, based on the key values at the bounds of the search space and the value of the sought key, usually via a linear interpolation
. The key value actually found at this estimated position is then compared to the key value being sought
. If it is not equal, then depending on the comparison, the remaining search space is reduced to the part before or after the estimated position
. This method will only work if calculations on the size of differences between key values are sensible""")

    def algo():
        print("""
Algorithm

Step 1 − Start searching data from middle of the list.

Step 2 − If it is a match, return the index of the item, and exit.

Step 3 − If it is not a match, probe position.

Step 4 − Divide the list using probing formula and find the new midle.

Step 5 − If data is greater than middle, search in higher sub-list.

Step 6 − If data is smaller than middle, search in lower sub-list.

Step 7 − Repeat until match.

probing formula:

mid = Lo + ((Hi - Lo) / (A[Hi] - A[Lo])) * (X - A[Lo])

where -
   A    = list
   Lo   = Lowest index of the list
   Hi   = Highest index of the list
   A[n] = Value stored at index n in the list
		""")


class exponential:

    def binarySearch(self, arr, l, r, x):
        if r >= l:
            mid = l + (r-l) // 2

            if arr[mid] == x:
                return mid

            if arr[mid] > x:
                return self.binarySearch(arr, l,
                                         mid - 1, x)

            return self.binarySearch(arr, mid + 1, r, x)
        return "element not found"

    def search(self, arr, n, x):
        if arr[0] == x:
            return 0
        i = 1
        while i < n and arr[i] <= x:
            i = i * 2

        return self.binarySearch(arr, i // 2,
                                 min(i, n-1), x)

    def show():
        print("""
def binarySearch( arr, l, r, x):
    if r >= l:
        mid = l + ( r-l ) // 2
        if arr[mid] == x:
            return mid
        if arr[mid] > x:
            return binarySearch(arr, l,mid - 1, x)
        return binarySearch(arr, mid + 1, r, x)

    return "element not found"
 def exponentialSearch(arr, n, x):
    if arr[0] == x:
        return 0
    i = 1
    while i < n and arr[i] <= x:
        i = i * 2
    return binarySearch( arr, i // 2,min(i, n-1), x) """)

    def info():
        print("""
 exponential search (also called doubling search or galloping search or Struzik search)is an algorithm, created by Jon Bentley and Andrew Chi-Chih Yao in 1976, for searching sorted, unbounded/infinite lists
.There are numerous ways to implement this with the most common being to determine a range that the search key resides in and performing a binary search within that range
.This takes O(log i) where i is the position of the search key in the list, if the search key is in the list, or the position where the search key should be, if the search key is not in the list""")

    def algo():
        print("""
Algorithm

consider example ExponentialSearch([1,2,3,4,5,6,7,8],3)

Step 1 − Checking whether the first element in the list matches the value we are searching for - since lys[0] is 1 and we are searching for 3, we set the index to 1 and move on.

Step 2 − Going through all the elements in the list, and while the item at the index'th position is less than or equal to our value, exponentially increasing the value of index in multiples of two:
	index = 1, lys[1] is 2, which is less than 3, so the index is multiplied by 2 and set to 2.
	index = 2, lys[2] is 3, which is equal to 3, so the index is multiplied by 2 and set to 4.
	index = 4, lys[4] is 5, which is greater than 3; the loop is broken at this point.

Step 3 − It then performs a binary search by slicing the list; arr[:4]
		""")


class ternary:

    def search(self, arr, x, l, r):
        if (r >= l):
            mid1 = l + (r - l)//3
            mid2 = mid1 + (r - l)//3

            if arr[mid1] == x:
                return mid1

            if arr[mid2] == x:
                return mid2

            if arr[mid1] > x:
                return self.search(arr, x, l, mid1-1)

            if arr[mid2] < x:
                return self.search(arr, x, mid2+1, r)

            return self.search(arr, x, mid1+1, mid2-1)

        return "element not found"

    def show():
        print("""
def ternarySearch(arr, l, r, x):
    if (r >= l):
        mid1 = l + (r - l)//3
        mid2 = mid1 + (r - l)//3
        if arr[mid1] == x:
            return mid1
        if arr[mid2] == x:
            return mid2
        if arr[mid1] > x:
            return ternarySearch(arr, l, mid1-1, x)
        if arr[mid2] < x:
            return ternarySearch(arr, mid2+1, r, x)
        return ternarySearch(arr, mid1+1, mid2-1, x)
    return "element not found" """)

    def info():
        print("""
A ternary search algorithm is a technique in computer science for finding the minimum or maximum of a unimodal function
. A ternary search determines either that the minimum or maximum cannot be in the first third of the domain or that it cannot be in the last third of the domain, then repeats on the remaining two thirds
. A ternary search is an example of a divide and conquer algorithm (see search algorithm)""")

    def algo():
        print("""
Algorithm

Step 1: Divide the search space (initially, the list) in three parts (with two mid-points: mid1 and mid2)

Step 2: The target element is compared with the edge elements that is elements at location mid1, mid2 and the end of the search space. If element matches, go to step 3 else predict in which section the target element lies. The search space is reduced to 1/3rd. If the element is not in the list, go to step 4 or to step 1.

Step 3: Element found. Return index and exit.

Step 4: Element not found. Exit.
		""")


''' sorting algorithms '''


class selection:

    def sort(arr):
        size = len(arr)
        for step in range(size):
            min_idx = step
            for i in range(step + 1, size):
                if arr[i] < arr[min_idx]:
                    min_idx = i
            (arr[step], arr[min_idx]) = (arr[min_idx], arr[step])
        return arr

    def show():
        print("""
def selectionsort(arr):
    size = len(arr)
    for step in range(size):
        min_idx = step
        for i in range(step + 1, size):
            if arr[i] < arr[min_idx]:
                min_idx = i
        (arr[step], arr[min_idx]) = (arr[min_idx], arr[step])
    return arr """)

    def info():
        print("""
Selection sort is a simple sorting algorithm
. This sorting algorithm is an in-place comparison-based algorithm in which the list is divided into two parts, the sorted part at the left end and the unsorted part at the right end
. Initially, the sorted part is empty and the unsorted part is the entire list
. The smallest element is selected from the unsorted array and swapped with the leftmost element, and that element becomes a part of the sorted array""")

    def algo():
        print("""
Algorithm

Step 1 - Get the value of n which is the total size of the array

Step 2 - Partition the list into sorted and unsorted sections. The sorted section is initially empty while the unsorted section contains the entire list

Step 3 - Pick the minimum value from the unpartitioned section and placed it into the sorted section.

Step 4 - Repeat the process (n – 1) times until all of the elements in the list have been sorted.
		""")


class bubble:

    def sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr

    def show():
        print("""
def bubblesort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr """)

    def info():
        print("""
Bubble Sort is a sorting algorithm used to sort list items in ascending order by comparing two adjacent values
. If the first value is higher than second value, the first value takes the second value position, while second value takes the first value position
. If the first value is lower than the second value, then no swapping is done
. This process is repeated until all the values in a list have been compared and swapped if necessary
. Each iteration is usually called a pass
. The number of passes in a bubble sort is equal to the number of elements in a list minus one""")

    def algo():
        print("""
Algorithm

Step 1 - Get the total number of elements. Get the total number of items in the given list

Step 2 - Determine the number of outer passes (n – 1) to be done. Its length is list minus one

Step 3 - Perform inner passes (n – 1) times for outer pass 1. Get the first element value and compare it with the second value. If the second value is less than the first value, then swap the positions

Step 4 - Repeat step 3 passes until you reach the outer pass (n – 1). Get the next element in the list then repeat the process that was performed in step 3 until all the values have been placed in their correct ascending order.

Step 5 - Return the result when all passes have been done. Return the results of the sorted list
		""")


class insertion:

    def sort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i-1
            while j >= 0 and key < arr[j]:
                    arr[j + 1] = arr[j]
                    j -= 1
            arr[j + 1] = key
        return arr

    def show():
        print("""
def insertionsort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j] :
                arr[j + 1] = arr[j]
                j -= 1
        arr[j + 1] = key
    return arr """)

    def info():
        print("""
Insertion sort is a sorting algorithm in which the elements are transferred one at a time to the right position
. In other words, an insertion sort helps in building the final sorted list, one item at a time, with the movement of higher-ranked elements
. An insertion sort has the benefits of simplicity and low overhead""")

    def algo():
        print("""
Algorithm

Step 1 - The first element in the array is assumed to be sorted. Take the second element and store it separately in key.

Step 2 - Compare key with the first element. If the first element is greater than key, then key is placed in front of the first element.

Step 3 - Now, the first two elements are sorted.Take the third element and compare it with the elements on the left of it.

step 4 - Placed it just behind the element smaller than it. If there is no element smaller than it, then place it at the beginning of the array.

Step 5 - Similarly, place every unsorted element at its correct position.
		""")


class merge:

    def sort(self, arr):
        if len(arr) > 1:
            mid = len(arr)//2
            L = arr[:mid]
            R = arr[mid:]
            self.sort(L)
            self.sort(R)
            i = j = k = 0
            while i < len(L) and j < len(R):
                if L[i] < R[j]:
                    arr[k] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                k += 1
            while i < len(L):
                arr[k] = L[i]
                i += 1
                k += 1
            while j < len(R):
                arr[k] = R[j]
                j += 1
                k += 1
        return arr

    def show():
        print("""
def mergesort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        sort(L)
        sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr """)

    def info():
        print("""
Merge sort is similar to the quick sort algorithm as it uses the divide and conquer approach to sort the elements
. It is one of the most popular and efficient sorting algorithm
. It divides the given list into two equal halves, calls itself for the two halves and then merges the two sorted halves
. We have to define the merge() function to perform the merging
.The sub-lists are divided again and again into halves until the list cannot be divided further
. Then we combine the pair of one element lists into two-element lists, sorting them in the process
. The sorted two-element pairs is merged into the four-element lists, and so on until we get the sorted list""")

    def algo():
        print("""
Algorithm

step 1 - Declare two variables left and right to mark the extreme indices of the array.

step 2 - Left will be equal to 0 and the value of right will be equal to size-1, where size is the length of the given unsorted array.

step 3 - Find the middle point of this array by applying mid = (left + right) / 2.

step 4 - Call the function mergeSort by passing the arguments as (left, mid) and (mid + 1, rear).

step 5 - The above steps will repeat till left < right.

step 6 - Then we will call the merge function on two sub-arrays
		""")


class heap:

    def sort(arr):
        heap = []
        for element in arr:
            heappush(heap, element)
        ordered = []
        while heap:
            ordered.append(heappop(heap))
        return ordered

    def show():
        print("""
def heapsort(arr):
    def heapify(arr, n, i):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        if l < n and arr[largest] < arr[l]:
            largest = l
        if r < n and arr[largest] < arr[r]:
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
    n = len(arr)
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)
    return arr """)

    def info():
        print("""
heapsort is a comparison-based sorting algorithm
. Heapsort can be thought of as an improved selection sort: like selection sort, heapsort divides its input into a sorted and an unsorted region, and it iteratively shrinks the unsorted region by extracting the largest element from it and inserting it into the sorted region
. Unlike selection sort, heapsort does not waste time with a linear-time scan of the unsorted region; rather, heap sort maintains the unsorted region in a heap data structure to more quickly find the largest element in each step
. Although somewhat slower in practice on most machines than a well-implemented quicksort, it has the advantage of a more favorable worst-case O(n log n) runtime
. Heapsort is an in-place algorithm, but it is not a stable sort
. Heapsort was invented by J.W.J.Williams in 1964
. This was also the birth of the heap, presented already by Williams as a useful data structure in its own right
. In the same year, R.W.Floyd published an improved version that could sort an array in-place, continuing his earlier research into the treesort algorithm
""")

    def algo():
        print("""
Algorithm
Sorry buddy !, heap sort is too complex to explain in step wise without images
		""")


class quick:
    def sort(self, start, end, array):
        ''''
        :param int start: array starting i.e 0
        :param int end: array ending i.e len(arr)-1
        :param int array: pass the array
         '''
        def partition(start, end, array):
            pivot_index = start
            pivot = array[pivot_index]
            while start < end:
                while start < len(array) and array[start] <= pivot:
                    start += 1
                while array[end] > pivot:
                    end -= 1
                if(start < end):
                    array[start], array[end] = array[end], array[start]
            array[end], array[pivot_index] = array[pivot_index], array[end]
            return end
        if (start < end):
            p = partition(start, end, array)
            self.sort(start, p - 1, array)
            self.sort(p + 1, end, array)
        return array

    def show():
        print("""
def quicksort(self,start, end, array):
        def partition(start, end, array):
            pivot_index = start
            pivot = array[pivot_index]
            while start < end:
                while start < len(array) and array[start] <= pivot:
                    start += 1
                while array[end] > pivot:
                    end -= 1
                if(start < end):
                    array[start], array[end] = array[end], array[start]
            array[end], array[pivot_index] = array[pivot_index], array[end]
            return end
        if (start < end):
            p = partition(start, end, array)
            self.sort(start, p - 1, array)
            self.sort(p + 1, end, array)
        return array """)

    def info():
        print("""
Like Merge Sort, QuickSort is a Divide and Conquer algorithm
. It picks an element as pivot and partitions the given array around the picked pivot
. There are many different versions of quickSort that pick pivot in different ways
. Always pick first element as pivot
. Always pick last element as pivot (implemented below)Pick a random element as pivot
. Pick median as pivot
. The key process in quickSort is partition()
. Target of partitions is, given an array and an element x of array as pivot, put x at its correct position in sorted array and put all smaller elements (smaller than x) before x, and put all greater elements (greater than x) after x
. All this should be done in linear time""")

    def algo():
        print("""
Algorithm

step 1 - Select the Pivot Element
    There are different variations of quicksort where the pivot element is selected from different positions.
    Here, we will be selecting the rightmost element of the array as the pivot element.

step 2 - Rearrange the Array
    Now the elements of the array are rearranged so that elements that are smaller than
    the pivot are put on the left and the elements greater than the pivot are put on the right.
    Here's how we rearrange the array:
        1.A pointer is fixed at the pivot element. The pivot element is compared with the elements beginning from the first index.
        2.If the element is greater than the pivot element, a second pointer is set for that element.
        3.Now, pivot is compared with other elements. If an element smaller than the pivot element is reached,
          the smaller element is swapped with the greater element found earlier.
        4.Again, the process is repeated to set the next greater element as the second pointer.
          And, swap it with another smaller element.
        5.The process goes on until the second last element is reached
        6.Finally, the pivot element is swapped with the second pointer.

step 3 - Divide Subarrays
    Pivot elements are again chosen for the left and the right sub-parts separately. And, step 2 is repeated.
    The subarrays are divided until each subarray is formed of a single element. At this point, the array is already sorted.

		""")


class count:

    def sort(arr):
        '''
         This particular radix sort cannot be used to sort non-positive integers \n
         and also this algo cannot sort non-intgers \n
         can only be strictly used for positive integers.
        '''
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

    def show():
        print("""
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
    return array """)

    def info():
        print("""
Counting sort is a sorting algorithm that sorts the elements of an array by counting the number of occurrences of each unique element in the array
. The count is stored in an auxiliary array and the sorting is done by mapping the count as an index of the auxiliary array
. Counting sort is a sorting technique based on keys between a specific range
. It works by counting the number of objects having distinct key values (kind of hashing)
. Then doing some arithmetic to calculate the position of each object in the output sequence

""")

    def algo():
        print("""
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



""")


class radix:

    def sort(arr):
        '''
         This particular radix sort cannot be used to sort non-positive integers \n
         and also this algo cannot sort non-intgers \n
         can only be strictly used for positive integers.
        '''
        def countingSort(arr, exp1):
            n = len(arr)
            output = [0] * (n)
            count = [0] * (10)
            for i in range(0, n):
                index = arr[i] // exp1
                count[index % 10] += 1
            for i in range(1, 10):
                count[i] += count[i - 1]
            i = n - 1
            while i >= 0:
                index = arr[i] // exp1
                output[count[index % 10] - 1] = arr[i]
                count[index % 10] -= 1
                i -= 1
            i = 0
            for i in range(0, len(arr)):
                arr[i] = output[i]
            return arr
        max1 = max(arr)
        exp = 1
        while max1 / exp > 1:
            array = countingSort(arr, exp)
            exp *= 10
        return array

    def show():
        print("""
def radixsort(arr):
    def countingSort(arr, exp1):
        n = len(arr)
        output = [0] * (n)
        count = [0] * (10)
        for i in range(0, n):
            index = arr[i] // exp1
            count[index % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = arr[i] // exp1
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
        i = 0
        for i in range(0, len(arr)):
            arr[i] = output[i]
        return arr
    max1 = max(arr)
    exp = 1
    while max1 / exp > 1:
        array = countingSort(arr, exp)
        exp *= 10
    return array """)

    def info():
        print("""
Radix sort is a sorting algorithm that sorts the elements by first grouping the individual digits of the same place value
. Then, sort the elements according to their increasing/decreasing order
. Suppose, we have an array of 8 elements
. First, we will sort elements based on the value of the unit place
. Then, we will sort elements based on the value of the tenth place
. This process goes on until the last significant place
""")

    def algo():
        print("""
Algorithm

step 1 - Find the largest element in the array, i.e. max. Let X be the number of digits in max.
         X is calculated because we have to go through all the significant places of all elements.

step 2 - Now, go through each significant place one by one.
         Use any stable sorting technique to sort the digits at each significant place. We have used counting sort for this.
         Sort the elements based on the unit place digits (X=0).

step 3 - Now, sort the elements based on digits at tens place.

step 4 - Finally, sort the elements based on the digits at hundreds place.
""")


class bucket:

    def sort(input_list):
        def insertion_sort(bucket):
            for i in range(1, len(bucket)):
                var = bucket[i]
                j = i - 1
                while (j >= 0 and var < bucket[j]):
                    bucket[j + 1] = bucket[j]
                    j = j - 1
                bucket[j + 1] = var
        max_value = max(input_list)
        size = max_value/len(input_list)
        buckets_list = []
        for x in range(len(input_list)):
            buckets_list.append([])
        for i in range(len(input_list)):
            j = int(input_list[i] / size)
            if j != len(input_list):
                buckets_list[j].append(input_list[i])
            else:
                buckets_list[len(input_list) - 1].append(input_list[i])
        for z in range(len(input_list)):
            insertion_sort(buckets_list[z])
        final_output = []
        for x in range(len(input_list)):
            final_output = final_output + buckets_list[x]
        return final_output

    def show():
        print("""
def sort(input_list):
    def insertion_sort(bucket):
        for i in range (1, len (bucket)):
            var = bucket[i]
            j = i - 1
            while (j >= 0 and var < bucket[j]):
                bucket[j + 1] = bucket[j]
                j = j - 1
            bucket[j + 1] = var
    max_value = max(input_list)
    size = max_value/len(input_list)
    buckets_list= []
    for x in range(len(input_list)):
        buckets_list.append([])
    for i in range(len(input_list)):
        j = int (input_list[i] / size)
        if j != len (input_list):
            buckets_list[j].append(input_list[i])
        else:
            buckets_list[len(input_list) - 1].append(input_list[i])
    for z in range(len(input_list)):
        insertion_sort(buckets_list[z])
    final_output = []
    for x in range(len (input_list)):
        final_output = final_output + buckets_list[x]
    return final_output """)

    def info():
        print("""
Bucket sort is a sorting algorithm that separate the elements into multiple groups said to be buckets
. Elements in bucket sort are first uniformly divided into groups called buckets, and then they are sorted by any other sorting algorithm
. After that, elements are gathered in a sorted manner
. (or) Bucket Sort is a sorting algorithm that divides the unsorted array elements into several groups called buckets
. Each bucket is then sorted by using any of the suitable sorting algorithms or recursively applying the same bucket algorithm

""")

    def algo():
        print("""
Algorithm


consider example array as : [.42, .32, .33, .52, .37, .47, .51]

step 1 - Create an array of size 10. Each slot of this array is used as a bucket for storing elements.

step 2 - Insert elements into the buckets from the array. The elements are inserted according to the range of the bucket.
         In our example code, we have buckets each of ranges from 0 to 1, 1 to 2, 2 to 3,...... (n-1) to n.
         Suppose, an input element is .23 is taken. It is multiplied by size = 10 (ie. .23*10=2.3).
         Then, it is converted into an integer (ie. 2.3≈2). Finally, .23 is inserted into bucket-2.
         Similarly, .25 is also inserted into the same bucket. Everytime, the floor value of the floating point number is taken.
         If we take integer numbers as input, we have to divide it by the interval (10 here) to get the floor value.
         Similarly, other elements are inserted into their respective buckets.

step 3 - The elements of each bucket are sorted using any of the stable sorting algorithms.
         Here, we have used quicksort (inbuilt function).

step 4 - The elements from each bucket are gathered.
         It is done by iterating through the bucket and inserting an individual element into the original array in each cycle.
         The element from the bucket is erased once it is copied into the original array.

""")


class shell:

    def sort(arr):
        gap = len(arr) // 2
        while gap > 0:
            i = 0
            j = gap
            while j < len(arr):
                if arr[i] > arr[j]:
                    arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j += 1
                k = i
                while k - gap > -1:
                    if arr[k - gap] > arr[k]:
                        arr[k-gap], arr[k] = arr[k], arr[k-gap]
                    k -= 1
            gap //= 2
        return arr

    def show():
        print("""
def shellsort(arr):
    gap = len(arr) // 2
    while gap > 0:
        i = 0
        j = gap
        while j < len(arr):
            if arr[i] >arr[j]:
                arr[i],arr[j] = arr[j],arr[i]
            i += 1
            j += 1
            k = i
            while k - gap > -1:
                if arr[k - gap] > arr[k]:
                    arr[k-gap],arr[k] = arr[k],arr[k-gap]
                k -= 1
        gap //= 2
    return arr

""")

    def info():
        print("""
Shell sort is a generalized version of the insertion sort algorithm
. It first sorts elements that are far apart from each other and successively reduces the interval between the elements to be sorted
. ShellSort is mainly a variation of Insertion Sort
. In insertion sort, we move elements only one position ahead
. When an element has to be moved far ahead, many movements are involved
. The idea of shellSort is to allow exchange of far items
. In shellSort, we make the array h-sorted for a large value of h
. We keep reducing the value of h until it becomes 1
. An array is said to be h-sorted if all sublists of every h’th element is sorted


""")

    def algo():
        print("""
Algorithm


consider example array : [9,8,3,7,5,6,4,1]

step 1 -In the first loop, if the array size is N = 8 then, the elements lying at the interval
        of N/2 = 4 are compared and swapped if they are not in order.
        The 0th element is compared with the 4th element.
        If the 0th element is greater than the 4th one then, the 4th element is first stored in temp variable and the 0th element (ie. greater element) is stored in the 4th position and the element stored in temp is stored in the 0th position.

step 2 - In the second loop, an interval of N/4 = 8/4 = 2 is taken and again the elements lying at these intervals are sorted.

step 3 - The elements at 4th and 2nd position are compared.
         The elements at 2nd and 0th position are also compared. All the elements in the array lying at the current interval are compared.

step 4 - The same process goes on for remaining elements.
         Finally, when the interval is N/8 = 8/8 =1 then the array elements lying at the interval of 1 are sorted.
         The array is now completely sorted.
""")


class comb:

    def sort(arr):
        def getNextGap(gap):
            gap = (gap * 10)//13
            if gap < 1:
                return 1
            return gap

        n = len(arr)
        gap = n
        swapped = True
        while gap != 1 or swapped == 1:
            gap = getNextGap(gap)
            swapped = False
            for i in range(0, n-gap):
                if arr[i] > arr[i + gap]:
                    arr[i], arr[i + gap] = arr[i + gap], arr[i]
                    swapped = True
        return arr

    def show():
        print("""
def combsort(arr):
    def getNextGap(gap):
        gap = (gap * 10)//13
        if gap < 1:
            return 1
        return gap

    n = len(arr)
    gap = n
    swapped = True
    while gap !=1 or swapped == 1:
        gap = getNextGap(gap)
        swapped = False
        for i in range(0, n-gap):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap]=arr[i + gap], arr[i]
                swapped = True
    return arr
""")

    def info():
        print("""
Comb Sort is mainly an improvement over Bubble Sort
. Bubble sort always compares adjacent values
. So all inversions are removed one by one
. Comb Sort improves on Bubble Sort by using gap of size more than 1
. The gap starts with a large value and shrinks by a factor of 1.3 in every iteration until it reaches the value 1
. Thus Comb Sort removes more than one inversion counts with one swap and performs better than Bubble Sort
. (or) It is a comparison-based sorting algorithm that is mainly an improvement in bubble sort
. In bubble sort, there is a comparison between the adjacent elements to sort the given array
. So, in bubble sort, the gap size between the elements that are compared is 1
. Comb sort improves the bubble sort by using a gap of size more than 1
. The gap in the comb sort starts with the larger value and then shrinks by a factor of 1.3
. It means that after the completion of each phase, the gap is divided by the shrink factor 1.3
. The iteration continues until the gap is 1
.The shrink factor is found to be 1.3 by testing comb sort on 200,000 random lists
. Comb sort works better than the bubble sort, but its time complexity in average case and worst case remains O(n2)
""")

    def algo():
        print("""
Algorithm

Sorry buddy !, heap sort is too complex to explain in step wise without images

""")


class pigeonhole:

    def sort(arr):
        my_min = min(arr)
        my_max = max(arr)
        size = my_max - my_min + 1
        holes = [0] * size
        for x in arr:
            holes[x - my_min] += 1
        i = 0
        for count in range(size):
            while holes[count] > 0:
                holes[count] -= 1
                arr[i] = count + my_min
                i += 1
        return arr

    def show():
        print("""
def pigeonhole(arr):
    my_min = min(arr)
    my_max = max(arr)
    size = my_max - my_min + 1
    holes = [0] * size
    for x in arr:
        holes[x - my_min] += 1
    i = 0
    for count in range(size):
        while holes[count] > 0:
            holes[count] -= 1
            arr[i] = count + my_min
            i += 1
    return arr
""")

    def info():
        print("""
Pigeonhole sorting is a sorting algorithm
.that is suitable for sorting lists of elements where the number of elements and the number of possible key values are
 approximately the same Where key here refers to the part of a group of data by which it is sorted, indexed, cross referenced, etc
.It is used in quick, heap and many other sorting algorithm
.If we consider number of elements (n) and the length of the range of possible key values (N) are approximately the same
.It requires O(n + N) time
""")

    def algo():
        print("""
Algorithm

step 1 - Iterate through the given list to find the least and greatest values in the array ‘a’.

step 2 - Let least element = ‘mn’ and greatest element b = ‘mx’.

step 3 - Let the range of possible values = ‘ mx+mn-1 ‘.

step 4 - Declare an array that is initialized with null pigeonholes the same size as of the range. Let array be named as ‘pihole’.

step 5 - Iterate through array again. Put each element in its Pigeonhole.

step 6 - An element at position a[i] in the array is put in the hole at index a[i] – mn.

step 7 - Now iterate over Pigeonhole array ie ‘pinhole’ array and put elements back into original array ‘a’.
""")


class cycle:
    def sort(array):
        writes = 0
        for cycleStart in range(0, len(array) - 1):
            item = array[cycleStart]
            pos = cycleStart
            for i in range(cycleStart + 1, len(array)):
                if array[i] < item:
                    pos += 1
            if pos == cycleStart:
                continue
            while item == array[pos]:
                pos += 1
            array[pos], item = item, array[pos]
            writes += 1
            while pos != cycleStart:
                pos = cycleStart
                for i in range(cycleStart + 1, len(array)):
                    if array[i] < item:
                        pos += 1
                while item == array[pos]:
                    pos += 1
                array[pos], item = item, array[pos]
                writes += 1
        return array

    
    def show():
        print("""
def cyclesort(array):
    writes = 0
    for cycleStart in range(0, len(array) - 1):
        item = array[cycleStart]
        pos = cycleStart
        for i in range(cycleStart + 1, len(array)):
            if array[i] < item:
                pos += 1
        if pos == cycleStart:
            continue
        while item == array[pos]:
            pos += 1
        array[pos], item = item, array[pos]
        writes += 1
        while pos != cycleStart:
            pos = cycleStart
            for i in range(cycleStart + 1, len(array)):
                if array[i] < item:
                    pos += 1
            while item == array[pos]:
                pos += 1
            array[pos], item = item, array[pos]
            writes += 1
    return array
""")



    def info():
        print("""
Cycle sort is an in-place, unstable sorting algorithm, a comparison sort that is theoretically optimal in terms of the total number of writes to the original array, unlike any other in-place sorting algorithm
. It is based on the idea that the permutation to be sorted can be factored into cycles, which can individually be rotated to give a sorted result
. It is optimal in terms of number of memory writes
. It minimizes the number of memory writes to sort (Each value is either written zero times, if it’s already in its correct position, or written one time to its correct position
. It is based on the idea that array to be sorted can be divided into cycles
. Cycles can be visualized as a graph
. We have n nodes and an edge directed from node i to node j if the element at i-th index must be present at j-th index in the sorted array
""")

    def algo():
        print("""
Algorithm

step 1 - Create a loop that begins at the beginning of the array and ends at the second to last item in the array

step 2 - Save the value of the item at the current index.

step 3 - In our example, we named our value item.

step 4 - Make a copy of the current index.

step 5 - In our example, we named our index copy currentIndexCopy.

step 6 - Create an additional loop that begins at one index after the currentIndex and ends at the last item in the array. Inside of this loop, compare item to the value of the item at the index of the loop we’re currently in. If the value at the index of the child loop is less than item, increment the currentIndexCopy.

step 7 - Once the loop from step 4 is completed, check to see if the currentIndexCopy has changed. If it has not changed, take a step forward in the loop.

step 8 - If the value at currentIndexCopy is the same as item, increment currentIndexCopy. This will skip all duplicate values.

step 9 - Save the item at the currentIndexCopy (we called our temp), place item into the index of currentIndexCopy, and update the value of item to temp.

step 10 - Create an additional loop that runs until currentIndex and currentIndexCopy are the same.

step 1 - Inside the loop, save currentIndex to currentIndexCopy.

step 1 - Repeat steps 4, 6 and 7.
""")




def checkprime( n, k=4):
        def miillerTest(d, n):
            def power(x, y, p):
                res = 1;	
                x = x % p;
                while (y > 0):
                    if (y & 1):
                        res = (res * x) % p;
                    y = y>>1; 
                    x = (x * x) % p;
                
                return res;
            a = 2 + random.randint(1, n - 4);
            x = power(a, d, n);
            if (x == 1 or x == n - 1):
                return True;
            while (d != n - 1):
                x = (x * x) % n;
                d *= 2;

                if (x == 1):
                    return False;
                if (x == n - 1):
                    return True;
            return False;
         
        if (n <= 1 or n == 4):
            return False;
        if (n <= 3):
            return True;

        d = n - 1;
        while (d % 2 == 0):
            d //= 2;

        for i in range(k):
            if (miillerTest(d, n) == False):
                return False;

        return True;

def showsieves():
        print("""
def getupto(n):
    primearr=[]
    prime = [True for i in range(n + 1)]
    p = 2
    while (p * p <= n):            
        if (prime[p] == True):                
            for i in range(p ** 2, n + 1, p):
                prime[i] = False
        p += 1
    prime[0]= False
    prime[1]= False
    for p in range(n + 1):
        if prime[p]:
            primearr.append(p)
    return primearr
    """)




class getprimes:

    def upto(n):
        primearr=[]
        prime = [True for i in range(n + 1)]
        p = 2
        while (p * p <= n):            
            if (prime[p] == True):                
                for i in range(p ** 2, n + 1, p):
                    prime[i] = False
            p += 1
        prime[0]= False
        prime[1]= False
        for p in range(n + 1):
            if prime[p]:
                primearr.append(p)
        return primearr
    
    def inrange(low, high):
        primearr=[]
        def fillPrimes(chprime, high):   
            ck = [True]*(high+1)
            l = int(math.sqrt(high))
            for i in range(2, l+1):
                if ck[i]:
                    for j in range(i*i, l+1, i):
                        ck[j] = False
            for k in range(2, l+1):
                if ck[k]:
                    chprime.append(k)
        
        chprime = list()
        fillPrimes(chprime, high)

        prime = [True] * (high-low + 1)
        for i in chprime:
            lower = (low//i)
            if lower <= 1:
                lower = i+i
            elif (low % i) != 0:
                lower = (lower * i) + i
            else:
                lower = lower*i
            for j in range(lower, high+1, i):
                prime[j-low] = False
                    
                    
        for k in range(low, high + 1):
                if prime[k-low]:
                    primearr.append(k)
        return primearr
            
class fib:
    def getelement(n):
        """
        Find the n'th fibonacci number. Uses memoization.

        :param n: the n'th fibonacci number to find
        :param m: dictionary used to store previous numbers
        :return: the value of the n'th fibonacci number
        """

        a = 0
        b = 1
        if n < 0:
            print("Incorrect input")
        elif n == 0:
            return a
        elif n == 1:
            return b
        else:
            for i in range(2, n):
                c = a + b
                a = b
                b = c
        return b
    
    def generate(n):
        cache = {0: 0, 1: 1}
        def fibonacci_of(n):
            if n in cache:
                return cache[n]
            cache[n] = fibonacci_of(n - 1) + fibonacci_of(n - 2)
            return cache[n]
        return [fibonacci_of(n) for n in range(n)]
        
class graph:
     
    def __init__(self): 
        self.graph = defaultdict(list)
        self.Time = 0
 
    def buildedge(self,u,v):
        self.graph[u].append(v)
    
    def buildmultiedge(self,coord_arr):
        self.V = len(coord_arr)//2
        for i in range(len(coord_arr)//2):
            self.buildedge(coord_arr[2*i],coord_arr[2*i+1]) 
 
    def BFS(self, s): 
        bfsarr = []
        visited = [False] * (max(self.graph) + 1) 
        queue = [] 
        queue.append(s)
        visited[s] = True
 
        while queue:
            s = queue.pop(0)
            bfsarr.append('-->')
            bfsarr.append(s)
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        return bfsarr
                    
 
    def dfshelper(self, v, visited):     
        visited.add(v)
        dfsarr.append('-->')
        dfsarr.append(v)
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.dfshelper(neighbour, visited) 
    def DFS(self, v):
        global dfsarr
        dfsarr = [] 
        visited = set() 
        self.dfshelper(v, visited)
        return dfsarr
    
    
    def APUtil(self, u, visited, ap, parent, low, disc):     
        children = 0 
        visited[u]= True 
        disc[u] = self.Time
        low[u] = self.Time
        self.Time += 1 
        for v in self.graph[u]:
            if visited[v] == False :
                parent[v] = u
                children += 1
                self.APUtil(v, visited, ap, parent, low, disc) 
                low[u] = min(low[u], low[v]) 
                if parent[u] == -1 and children > 1:
                    ap[u] = True 
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap[u] = True                        
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    def findAP(self):  
        visited = [False] * (self.V)
        disc = [float("Inf")] * (self.V)
        low = [float("Inf")] * (self.V)
        parent = [-1] * (self.V)
        ap = [False] * (self.V) 
        for i in range(self.V):
            if visited[i] == False:
                self.APUtil(i, visited, ap, parent, low, disc)
        for index, value in enumerate (ap):
            if value == True: print (index,end=" ")
            
class pattern:
    global KMPSearch
    def KMPSearch(pat, txt):
        def computeLPSArray(pat, M, lps):
            len = 0 
            lps[0] 
            i = 1
            while i < M:
                if pat[i]== pat[len]:
                    len += 1
                    lps[i] = len
                    i += 1
                else:
                    if len != 0:
                        len = lps[len-1]
        
                    else:
                        lps[i] = 0
                        i += 1
            return pat, M, lps
        M = len(pat)
        N = len(txt)
        lps = [0]*M
        j = 0 
        pat, M, lps = computeLPSArray(pat, M, lps)
        i = 0 
        while i < N:
            if pat[j] == txt[i]:
                i += 1
                j += 1
            if j == M:
                found = True
                index = i-j
                break
                j = lps[j-1]
            elif i < N and pat[j] != txt[i]:
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1
            else:
                found = False
                index = 'not found'
           
        return found,index
  
    class isthere:
        def __init__(self,a):
            self.a = a
        def inn(self,b):
            a,b = KMPSearch(self.a, b)
            return a
        
    class whereis:
        def __init__(self,c):
            self.c = c
        def inn(self,b):
            a,x = KMPSearch(self.c, b)
            return x
            
class getprimefactors:
    def fornum(n):  
        primefactorsarr = []       
        while n % 2 == 0:
            primefactorsarr.append(2)
            n=n/2           
        for i in range(3,int(math.sqrt(n))+1,2):            
            while n % i== 0:
                primefactorsarr.append(i)    
                n=n/i            
        if n > 2:
            primefactorsarr.append(n)
        return primefactorsarr
                      
def findgcdof(a, b):
    if a == 0 :
        return b
     
    return findgcdof(b%a, a)

class findinversions:
    def forr(arr):
        N = len(arr)
        if N <= 1:
            return 0
        sortList = []
        result = 0    
        for i, v in enumerate(arr):
            heappush(sortList, (v, i))
        x = [] 
        while sortList: 
            v, i = heappop(sortList)              
            y = bisect(x, i) 
            result += i - y
            insort(x, i)  
        return result

class catlan_mumbers:
    def getelement(x):
        def catalan(n):
            def binomialCoefficient(n, k):     
                if (k > n - k):
                    k = n - k        
                res = 1        
                for i in range(k):
                    res = res * (n - i)
                    res = res / (i + 1)
                return res
            c = binomialCoefficient(2*n, n)
            return c/(n + 1)
        return catalan(x)
    def gen(x):
        catlanarr= []
        def catalan(n):
            def binomialCoefficient(n, k):     
                if (k > n - k):
                    k = n - k        
                res = 1        
                for i in range(k):
                    res = res * (n - i)
                    res = res / (i + 1)
                return res
            c = binomialCoefficient(2*n, n)
            return c/(n + 1)
        for i in range(x):
            catlanarr.append(catalan(i))
        return catlanarr

class istheresum:
    def __init__(self,number):
        self.n = number
    def inarr(self,arr):
        def isSubsetSum(arr, n, summ):
            if (summ == 0):
                return True
            if (n == 0):
                return False
            if (arr[n - 1] > summ):
                return isSubsetSum(arr, n - 1, summ)
            return isSubsetSum(
                arr, n-1, summ) or isSubsetSum(
                arr, n-1, summ-arr[n-1])
        return isSubsetSum(arr, len(arr), self.n)