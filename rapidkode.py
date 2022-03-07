import math


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

    def search(self, arr,x, l, r):
        if (r >= l):
            mid1 = l + (r - l)//3
            mid2 = mid1 + (r - l)//3

            if arr[mid1] == x:
                return mid1

            if arr[mid2] == x:
                return mid2

            if arr[mid1] > x:
                return self.search(arr,x, l, mid1-1)

            if arr[mid2] < x:
                return self.search(arr, x,mid2+1, r)

            return self.search(arr, x,mid1+1, mid2-1)

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


''' Searching algorithms '''


