## LeetCode 编程练习

## LeetCode 34. Search for a Range

### Description
Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

For example,
Given [5, 7, 7, 8, 8, 10] and target value 8,
return [3, 4].

``` java
class Solution {
    // 返回最左边或这最右边等于 target 的值在数组中的下标。
    private int extremeInsertionIndex(int[] nums, int target, boolean left) {
        int lo = 0;
        int hi = nums.length;

        while (lo < hi) {
            int mid = (lo+hi)/2;
            if (nums[mid] > target || (left && target == nums[mid])) {
                hi = mid;
            }
            else {
                lo = mid+1;
            }
        }
        
        return lo;
    }

    public int[] searchRange(int[] nums, int target) {
        int[] targetRange = {-1, -1};

        int leftIdx = extremeInsertionIndex(nums, target, true);
		
		// 断言 leftIdx 在数组范围内并且 target 在数组中
        if (leftIdx == nums.length || nums[leftIdx] != target) {
            return targetRange;
        }

        targetRange[0] = leftIdx;
        targetRange[1] = extremeInsertionIndex(nums, target, false)-1;

        return targetRange;
    }
}
```
### Complexity Analysis

Time complexity : O(lgn)
728. Self Dividing Numbers
Because binary search cuts the search space roughly in half on each iteration, there can be at most \lceil lgn \rceil⌈lgn⌉ iterations. Binary search is invoked twice, so the overall complexity is logarithmic.

Space complexity : O(1)

All work is done in place, so the overall memory usage is constant.



## LeetCode 728. Self Dividing Numbers
### Description
A self-dividing number is a number that is divisible by every digit it contains.

For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.

Also, a self-dividing number is not allowed to contain the digit zero.

Given a lower and upper number bound, output a list of every possible self dividing number, including the bounds if possible.

``` java
class Solution {
    public List<Integer> selfDividingNumbers(int left, int right) {
        List<Integer> ret = new ArrayList<>();
        for (int n = left; n <= right; ++n) {
            if (selfDividing(n)) {
                ret.add(n);
            }
        }
        return ret;
    }
    public boolean selfDividing(int n) {
        for (char c: String.valueOf(n).toCharArray()) {
            if (c == '0' || (n % (c - '0') > 0))
                return false;
        }
        return true;
    }
}
```
### Complexity Analysis

Time Complexity: O(D), where D is the number of integers in the range [L,R], and assuming log(R) is bounded. (In general, the complexity would be O(DlogR).)

Space Complexity: O(D), the length of the answer.

## LeetCode 26. Remove Duplicates from Sorted Array

### Description
Given a sorted array, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

``` java
class Solution {
    public int removeDuplicates(int[] nums) {
        int count = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[count] != nums[i]) {
                count++;
                nums[count] = nums[i];
            }
        }
        return count + 1;
    }
}
```
### Complexity analysis
Time complextiy : O(n). Assume that n is the length of array. Each of i and j traverses at most n steps.

Space complexity : O(1).


## LeetCode 27. Remove Element

### Description
Given an array and a value, remove all instances of that value in-place and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

The order of elements can be changed. It doesn't matter what you leave beyond the new length.

``` java
class Solution {
    public int removeElement(int[] nums, int val) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[j] != val) {
                nums[i++] = nums[j];
            }
        }

        return i;
    }
}
```

### Complexity analysis

Time complexity : O(n). Assume the array has a total of n elements, both i and j traverse at most 2n steps.

Space complexity : O(1).

## LeetCode 80. Remove Duplicates from Sorted Array II

### Description
Follow up for "Remove Duplicates":
What if duplicates are allowed at most twice?

For example,
Given sorted array nums = [1,1,1,2,2,3],

Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3. It doesn't matter what you leave beyond the new length.

``` java
class Solution {
    public int removeDuplicates(int[] nums) {
        int i = 0;
        for (int n: nums) {
            if (i < 2 || n > nums[i - 2])
                nums[i++] = n;
        }

        return i;
    }
}
```

## LeetCode 75. Sort Colors

### Description
Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

#### Note:
You are not suppose to use the library's sort function for this problem.

``` java
class Solution {
    public void sortColors(int[] nums) {
        int[] count = new int[3];
        for (int i = 0; i < nums.length; i++) {
            count[nums[i]]++;
        }
        int k = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < count[i]; j++) {
                nums[k++] = i;
            }
        }
    }
}
```

## LeetCode 88. Merge Sorted Array

### Description

Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

#### Note:
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.

``` java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;
        while (i >= 0 && j >= 0) {
            if (nums1[i] > nums2[j]) {
                nums1[k--] = nums1[i--];
            }
            else {
                nums1[k--] = nums2[j--];
            }
        }
        while (j >= 0) {
            nums1[k--] = nums2[j--];
        }
    }
}
```
## LeetCode 215. Kth Largest Element in an Array

### Description
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

For example,
Given [3,2,1,5,6,4] and k = 2, return 5.

#### Note: 
You may assume k is always valid, 1 ≤ k ≤ array's length.

``` java

    public class Solution {
        public int findKthLargest(int[] nums, int k) {
            int index = quickSelect(nums, 0, nums.length-1, k-1);
            return nums[index];
        }

        private int quickSelect(int[] nums, int lo, int hi, int k) {
            int lt = lo;
            int gt = hi + 1;
            int i = lo + 1;
            int pivot = nums[lo];

            while (i < gt) {
                if (nums[i] > pivot) {
                    swap(nums, i, lt + 1);
                    lt++;
                    i++;
                }
                else if (nums[i] < pivot) {
                    swap(nums, i, gt - 1);
                    gt--;
                }
                else {
                    i++;
                }
            }

            swap(nums, lo, lt);

            if (lt == k) {
                return lt;
            }
            else if (lt < k) {
                return quickSelect(nums, lt + 1, hi, k);
            }
            else {
                return quickSelect(nums, lo, lt - 1, k);
            }
        }


        private void swap(int[] a, int i, int j) {
            int temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }
```
### Complexity analysis

Time complexity : O(n). 
Space complexity : O(log(n

## LeetCode 167. Two Sum II - Input array is sorted

### Description
Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.

You may assume that each input would have exactly one solution and you may not use the same element twice.

Input: numbers={2, 7, 11, 15}, target=9
Output: index1=1, index2=2

``` java	
    class Solution {
        public int[] twoSum(int[] numbers, int target) {
            int[] ans = new int[2];
            int lo = 0;
            int hi = numbers.length-1;
            int sum;
            while (lo < hi) {
                sum = numbers[lo] + numbers[hi];
                if(sum < target) {
                    lo++;
                }
                else if (sum > target) {
                    hi--;
                }
                else {
                    ans[0] = lo + 1;
                    ans[1] = hi + 1;
                    break;
                }
            }

            return ans;
        }
    }
```

## LeetCode 125. Valid Palindrome
### Description
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring other cases.

For example,
"A man, a plan, a canal: Panama" is a palindrome.
"race a car" is not a palindrome.

#### Note:
Have you consider that the string might be empty? This is a good question to ask during an interview.

For the purpose of this problem, we define empty string as valid palindrome.

``` java
class Solution {
        public boolean isPalindrome(String s) {
            if (s.isEmpty()) {
                return true;
            }
            int lo = 0;
            int hi = s.length() - 1;
            char clo, chi;
            while (lo <= hi) {
                clo = s.charAt(lo);
                chi = s.charAt(hi);
                if (!Character.isLetterOrDigit(clo)) {
                    lo++;
                }
                else if (!Character.isLetterOrDigit(chi)) {
                    hi--;
                }
                else {
                    if (Character.toLowerCase(clo) != Character.toLowerCase(chi)) {
                        return false;
                    }
                    lo++;
                    hi--;
                }
            }

            return true;
        }
    }
```

## LeetCode 344. Reverse String

### Description
Write a function that takes a string as input and returns the string reversed.

Example:
Given s = "hello", return "olleh".

``` java
    class Solution {
        public String reverseString(String s) {
            char[] str = s.toCharArray();
            int h = s.length() - 1;
            int l = 0;

            while (l < h) {
                char tmp = str[l];
                str[l] = str[h];
                str[h] = tmp;
                l++;
                h--;
            }
            return new String(str);
        }
    }
```

### Complexity Analysis

Time Complexity: `O(n)` (Average Case) and `O(n)` (Worst Case) where `n` is the total number character in the input string. The algorithm need to reverse the whole string.

Auxiliary Space: `O(n)` space is used where `n` is the total number character in the input string. Space is needed to transform string to character array.


## LeetCode 345. Reverse Vowels of a String

### Description
Write a function that takes a string as input and reverse only the vowels of a string.

#### Example 1:
Given s = "hello", return "holle".

#### Example 2:
Given s = "leetcode", return "leotcede".

``` java
    class Solution {
        public String reverseVowels(String s) {
            char[] str = s.toCharArray();
            String vol = "aeiouAEIOU";
            int lo = 0;
            int hi = str.length - 1;
            while (lo < hi) {
                while (lo < hi && !vol.contains(str[lo]+"")) {
                    lo++;
                }
                while (lo < hi && !vol.contains(str[hi] + "")) {
                    hi--;
                }
                char tmp = str[lo];
                str[lo] = str[hi];
                str[hi] = tmp;
                lo++;
                hi--;
            }
            return new String(str);
        }
    }
```
-

## LeetCode 20. Valid Parentheses
### Description
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.

``` java
   class Solution {
        public boolean isValid(String s) {
            Stack<Character> stack = new Stack<>();
            for (char c : s.toCharArray()) {
                if (c == '(') {
                    stack.push(')');
                }
                else if (c == '{') {
                    stack.push('}');
                }
                else if (c == '[') {
                    stack.push(']');
                }
                else if (stack.isEmpty() || stack.pop() != c) {
                    return false;
                }
            }
           return stack.isEmpty();
        }
```


## LeetCode 9. Palindrome Number
### Description
Determine whether an integer is a palindrome. Do this without extra space.

``` java
    class Solution {
        public boolean isPalindrome(int x) {
            String s = Integer.toString(x);
            int i = 0;
            int j = s.length() - 1;
            char[] chars = s.toCharArray();
            while (i < j) {
                if (chars[i] != chars[j]) {
                    return false;
                }
                else {
                    i++;
                    j--;
                }
            }
            return true;
        }
    }
```

## leetCode 3. Longest Substring Without Repeating Characters
### Description
Given a string, find the length of the longest substring without repeating characters.

##### Examples:

Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

``` java
    class Solution {
        public int lengthOfLongestSubstring(String s) {
            int n = s.length();
            Set<Character> set = new HashSet<>();
            int ans = 0, i = 0, j = 0;
            while (i < n && j < n) {
                if (!set.contains(s.charAt(j))) {
                    set.add(s.charAt(j++));
                    ans = Math.max(ans, j - i);
                }
                else {
                    set.remove(s.charAt(i++));
                }
            }
            return ans;
        }
    }
	
	public class Solution {
    public int lengthOfLongestSubstring(String s) {
        int n = s.length(), ans = 0;
        int[] index = new int[128]; // current index of character
        // try to extend the range [i, j]
        for (int j = 0, i = 0; j < n; j++) {
            i = Math.max(index[s.charAt(j)], i);
            ans = Math.max(ans, j - i + 1);
            index[s.charAt(j)] = j + 1;
        }
        return ans;
    }
}
```

## LeetCode 438. Find All Anagrams in a String
### Description
Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

The order of output does not matter

``` java
class Solution {
        public List<Integer> findAnagrams(String s, String p) {
            List<Integer> list = new ArrayList<>();
            if (s == null || s.length() == 0 || p == null || p.length() == 0) {
                return list;
            }
            int[] hash = new int[256];
            for (char c : p.toCharArray()) {
                hash[c]++;
            }

            int lo = 0;
            int hi = 0;
            int count = p.length();

            while (hi < s.length()) {
                if (hash[s.charAt(hi)] >= 1) {
                    count--;
                }
                hash[s.charAt(hi)]--;
                hi++;
                if (count == 0) {
                    list.add(lo);
                }
                if (hi - lo == p.length()) {
                    if (hash[s.charAt(lo)] >= 0) {
                        count++;
                    }
                    hash[s.charAt(lo)]++;
                    lo++;
                }
            }
            return list;
        }
    }
```


## LeetCode 209. Minimum Size Subarray Sum
### Description

Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

For example, given the array [2,3,1,2,4,3] and s = 7,
the subarray [4,3] has the minimal length under the problem constraint.

``` java
    class Solution {
        public int minSubArrayLen(int s, int[] nums) {
            int i = 0, j = -1;
            int ans = nums.length + 1;
            int sum = 0;
            while (i < nums.length) {
                if(sum < s && j + 1 < nums.length) {
                    sum += nums[++j];
                }
                else {
                    sum -= nums[i++];
                }
                if(sum >= s) {
                    ans = Math.min(ans, j - i + 1);
                }
            }
            if (ans == nums.length + 1)
                return 0;
            return ans;
        }
    }
```
## 76. Minimum Window Substring
### Description

Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

For example,
S = "ADOBECODEBANC"
T = "ABC"
Minimum window is "BANC".

Note:
If there is no such window in S that covers all characters in T, return the empty string "".

If there are multiple such windows, you are guaranteed that there will always be only one unique minimum window in S.

``` java
    class Solution {
        public String minWindow(String s, String t) {
            if (s.isEmpty() || t.isEmpty() || s.length() < t.length())
                return "";
            int minlen = s.length();
            int minstart = 0;
            int minend = s.length()-1;
            HashMap<Character, Integer> require = new HashMap<Character, Integer>();

            for (char c : t.toCharArray()) {
                require.put(c, require.containsKey(c) ? require.get(c) + 1 : 1);
            }

            int count = t.length();
            int li = 0;

            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);

                if (require.containsKey(c)) {
                    if (require.get(c) > 0)
                        count--;
                    require.put(c, require.get(c) - 1);
                }

                if (count == 0) {
                    char cli = s.charAt(li);
                    while (!require.containsKey(cli) || require.get(cli) < 0) {
                        if (require.containsKey(cli)) {
                            require.put(cli, require.get(cli) + 1);
                        }
                        li++;
                        cli = s.charAt(li);
                    }

                    if (minlen > i - li + 1) {
                        minstart = li;
                        minend = i;
                        minlen = i - li + 1;
                    }
                }
            }
            if (count != 0) {
                return "";
            }
            return s.substring(minstart, minend + 1);
        }
```

## 713. Subarray Product Less Than K

### Description

Your are given an array of positive integers nums.

Count and print the number of (contiguous) subarrays where the product of all the elements in the subarray is less than k.

##### Example 1:
```
Input: nums = [10, 5, 2, 6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are: [10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6].
Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.
```

##### Note:
```
0 < nums.length <= 50000.
0 < nums[i] < 1000.
0 <= k < 10^6.
```

```java
 class Solution {
        public int numSubarrayProductLessThanK(int[] nums, int k) {
            if (k <= 1)
                return 0;
            int prod = 1;
            int ans = 0;
            int left = 0;
            for (int right = 0; right < nums.length; right++) {
                prod *= nums[right];
                while (prod >= k) {
                    prod /= nums[left++];
                }
                ans += right - left + 1;
            }

            return ans;
        }
    }
```

## LeetCode 11. Container With Most Water

### Description
Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

``` java
    class Solution {
        public int maxArea(int[] height) {
            int lo = 0;
            int hi = height.length - 1;
            int maxarea = 0;

            while (lo < hi) {
                maxarea = Math.max(maxarea, Math.min(height[lo], height[hi]) * (hi - lo));
                if (height[lo] < height[hi]) {
                    lo++;
                }
                else
                    hi--;

            }
            return maxarea;
        }
    }
```

### Complexity Analysis

Time complexity : O(n). Single pass.

Space complexity : O(1). Constant space is used.


## 303. Range Sum Query - Immutable

### Description

Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.

##### Example:
```
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
```
##### Note:
You may assume that the array does not change.
There are many calls to sumRange function.

``` java
    class NumArray {
        private int[] sum;
        public NumArray(int[] nums) {
            sum = new int[nums.length + 1];
            for (int i = 0; i < nums.length; i++) {
                sum[i + 1] = sum[i] + nums[i];
            }
        }

        public int sumRange(int i, int j) {
            return sum[j + 1] - sum[i];
        }
    }
```
## 121. Best Time to Buy and Sell Stock
### Description

Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

##### Example 1:
```
Input: [7, 1, 5, 3, 6, 4]
Output: 5

max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)
```
##### Example 2:
```
Input: [7, 6, 4, 3, 1]
Output: 0

In this case, no transaction is done, i.e. max profit = 0.
```

``` java
    class Solution {
        public int maxProfit(int[] prices) {
            int min = Integer.MAX_VALUE;
            int max = 0;
            for (int i = 0; i < prices.length; i++) {
                if (prices[i] < min) {
                    min = prices[i];
                }
                else if (prices[i] - min > max) {
                    max = prices[i] - min;
                }
            }
            return max;
        }
    }
```

## 122. Best Time to Buy and Sell Stock II
### Description

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

``` java
    class Solution {
        public int maxProfit(int[] prices) {
            int sum = 0;
            for (int i = 1; i < prices.length; i++) {
                if (prices[i] - prices[i-1]>= 1) {
                    sum += prices[i] - prices[i-1];
                }
            }
            return sum;
        }
    }
```

# DP ????
## 123. Best Time to Buy and Sell Stock III
### Description

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most two transactions.

Note:
You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

``` java
enter code here
```



## 717. 1-bit and 2-bit Characters

### Description

We have two special characters. The first character can be represented by one bit 0. The second character can be represented by two bits (10 or 11).

Now given a string represented by several bits. Return whether the last character must be a one-bit character or not. The given string will always end with a zero.

##### Example 1:
```
Input: 
bits = [1, 0, 0]
Output: True
Explanation: 
The only way to decode it is two-bit character and one-bit character. So the last character is one-bit character.
```
##### Example 2:
```
Input: 
bits = [1, 1, 1, 0]
Output: False
Explanation: 
The only way to decode it is two-bit character and two-bit character. So the last character is NOT one-bit character.
```
##### Note:

- 1 <= len(bits) <= 1000.
- bits[i] is always 0 or 1.


``` java
class Solution {
    public boolean isOneBitCharacter(int[] bits) {
        int i = 0;
        while (i < bits.length - 1) {
            i += bits[i] + 1;
        }
        return i == bits.length - 1;
    }
}
```

## 647. Palindromic Substrings
### Description

Given a string, your task is to count how many palindromic substrings in this string.

The substrings with different start indexes or end indexes are counted as different substrings even they consist of same characters.

##### Example 1:
Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".

##### Example 2:
Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".

##### Note:
The input string length won't exceed 1000.

``` java
    class Solution {

        int count = 0;

        public int countSubstrings(String s) {

            if (s == null || s.length() == 0)
                return 0;

            for (int i = 0; i < s.length(); i ++) {
                extendPalindrome(s, i, i);
                extendPalindrome(s, i, i + 1);
            }
			
			return count;
        }

        private void extendPalindrome(String s, int left, int right) {
            while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
                count++;
                left--;
                right++;
            }
        }
    }
```

## 268. Missing Number
### Description

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

##### Example 1
Input: [3,0,1]
Output: 2

##### Example 2
Input: [9,6,4,2,3,5,7,0,1]
Output: 8

##### Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?

``` java
    class Solution {
        public int missingNumber(int[] nums) {
            int sum = nums.length * (nums.length + 1) / 2;

            for (int i = 0; i < nums.length; i++) {
                sum -= nums[i];
            }

            return sum;
        }
    }
```

## 56. Merge Intervals
### Description

Given a collection of intervals, merge all overlapping intervals.

For example,
Given [1,3],[2,6],[8,10],[15,18],
return [1,6],[8,10],[15,18].

``` java
    class Solution {
        public List<Interval> merge(List<Interval> intervals) {

            int n = intervals.size();
            int[] starts = new int[n];
            int[] ends = new int[n];

            for (int i = 0; i < n; i++) {
                starts[i] = intervals.get(i).start;
                ends[i] = intervals.get(i).end;
            }

            Arrays.sort(starts);
            Arrays.sort(ends);

            List<Interval> res = new ArrayList<>();
            for (int i = 0, j = 0; i < n; i++) {
                if (i == n - 1 || starts[i + 1] > ends[i]) {
                    res.add(new Interval(starts[j], ends[i]));
                    j  = i + 1;
                }
            }

            return res;
        }
    }
```


## 434. Number of Segments in a String

### Description
Count the number of segments in a string, where a segment is defined to be a contiguous sequence of non-space characters.

Please note that the string does not contain any non-printable characters.

##### Example:

Input: "Hello, my name is John"
Output: 5

``` java
    class Solution {
        public int countSegments(String s) {

            int count = 0;

            for (int i = 0; i < s.length(); i++) {
                if (s.charAt(i) != ' ' && (i == 0 || s.charAt(i - 1) == ' ')
                    count++;
            }

            return count;
        }
    }
```


## 387. First Unique Character in a String
### Description

Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

##### Examples:

s = "leetcode"
return 0.

s = "loveleetcode",
return 2.

##### Note: 
You may assume the string contain only lowercase letters.

``` java
    class Solution {
        public int firstUniqChar(String s) {

            int[] freq = new int[26];

            for (int i = 0; i < s.length(); i++) {
                freq[s.charAt(i) - 'a']++;
            }
            for (int i = 0; i < s.length(); i++) {
                if (freq[s.charAt(i) - 'a'] == 1)
                    return i;
            }
            return -1;
        }
    }
```

## 696. Count Binary Substrings
### Description
Give a string s, count the number of non-empty (contiguous) substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.

Substrings that occur multiple times are counted the number of times they occur.

##### Example 1:
Input: "00110011"
Output: 6

``` 
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
Notice that some of these substrings repeat and are counted the number of times they occur.
Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.
```


##### Example 2:

``` 
Input: "10101"
Output: 4
Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal number of consecutive 1's and 0's.
```
-

``` java
enter code here
```

## CCF 201709-2 公共钥匙盒 Java
### 问题描述 
　　有一个学校的老师共用N个教室，按照规定，所有的钥匙都必须放在公共钥匙盒里，老师不能带钥匙回家。每次老师上课前，都从公共钥匙盒里找到自己上课的教室的钥匙去开门，上完课后，再将钥匙放回到钥匙盒中。 
　　钥匙盒一共有N个挂钩，从左到右排成一排，用来挂N个教室的钥匙。一串钥匙没有固定的悬挂位置，但钥匙上有标识，所以老师们不会弄混钥匙。 
　　每次取钥匙的时候，老师们都会找到自己所需要的钥匙将其取走，而不会移动其他钥匙。每次还钥匙的时候，还钥匙的老师会找到最左边的空的挂钩，将钥匙挂在这个挂钩上。如果有多位老师还钥匙，则他们按钥匙编号从小到大的顺序还。如果同一时刻既有老师还钥匙又有老师取钥匙，则老师们会先将钥匙全还回去再取出。 
　　今天开始的时候钥匙是按编号从小到大的顺序放在钥匙盒里的。有K位老师要上课，给出每位老师所需要的钥匙、开始上课的时间和上课的时长，假设下课时间就是还钥匙时间，请问最终钥匙盒里面钥匙的顺序是怎样的？ 
输入格式 
　　输入的第一行包含两个整数N, K。 
　　接下来K行，每行三个整数w, s, c，分别表示一位老师要使用的钥匙编号、开始上课的时间和上课的时长。可能有多位老师使用同一把钥匙，但是老师使用钥匙的时间不会重叠。 
　　保证输入数据满足输入格式，你不用检查数据合法性。 
输出格式 
　　输出一行，包含N个整数，相邻整数间用一个空格分隔，依次表示每个挂钩上挂的钥匙编号。 
样例输入 
5 2 
4 3 3 
2 2 7 
样例输出 
1 4 3 2 5 
样例说明 
　　第一位老师从时刻3开始使用4号教室的钥匙，使用3单位时间，所以在时刻6还钥匙。第二位老师从时刻2开始使用钥匙，使用7单位时间，所以在时刻9还钥匙。 
　　每个关键时刻后的钥匙状态如下（X表示空）： 
　　时刻2后为1X345； 
　　时刻3后为1X3X5； 
　　时刻6后为143X5； 
　　时刻9后为14325。 
样例输入 
5 7 
1 1 14 
3 3 12 
1 15 12 
2 7 20 
3 18 12 
4 21 19 
5 30 9 
样例输出 
1 2 3 5 4 
评测用例规模与约定 
　　对于30%的评测用例，1 ≤ N, K ≤ 10, 1 ≤ w ≤ N, 1 ≤ s, c ≤ 30； 
　　对于60%的评测用例，1 ≤ N, K ≤ 50，1 ≤ w ≤ N，1 ≤ s ≤ 300，1 ≤ c ≤ 50； 
　　对于所有评测用例，1 ≤ N, K ≤ 1000，1 ≤ w ≤ N，1 ≤ s ≤ 10000，1 ≤ c ≤ 100。


```
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;

public class Main {
    private static class Node implements Comparable<Node>{
        private int key;
        private int time;
        private int isget;

        Node(int k, int t, int f) {
            key = k;
            time = t;
            isget = f;
        }
        @Override
        public int compareTo(Main.Node o) {
            if (time > o.time)
                return 1;
            else if (time == o.time) {
                if (isget > o.isget) {
                    return 1;
                }
                else if (isget == o.isget) {
                    if (key > o.key) {
                        return 1;
                    }
                    else if (key == o.key) {
                        return 0;
                    }
                    else
                        return -1;
                }
                else
                    return -1;
            }
            return -1;
        }

        @Override
        public String toString() {
            return "Node{" +
                    "key= " + key +
                    ", time= " + time +
                    ", isget= " + isget +
                    '}';
        }
    }

    public static void main(String[] args) {
        Scanner In = new Scanner(System.in);
        int N = In.nextInt();
        int K = In.nextInt();
        ArrayList<Node> nodes = new ArrayList<>();
        int[] map = new int[N+1];
        for (int i = 0; i <= N; i++) {
            map[i] = i;
        }
        int w, s, c;
        while (K -- > 0) {
            w = In.nextInt();
            s = In.nextInt();
            c = In.nextInt();
            nodes.add(new Node(w, s,1));
            nodes.add(new Node(w, s + c,0));
        }

        Collections.sort(nodes);
//        for (int i = 0; i < nodes.size(); i++) {
//            System.out.println(nodes.get(i).toString());
//        }

        for (Node node :nodes) {
            int isget = node.isget;
            if (isget == 1) {
                for (int i = 1; i <= N; i++) {
                    if (node.key == map[i]) {
                        map[i] = 0;
                        break;
                    }
                }
            }
            else {
                for (int i = 1; i <= N; i++) {
                    if (map[i] == 0) {
                        map[i] = node.key;
                        break;
                    }
                }
            }
        }
        boolean flag = true;
        for (int i = 1; i <= N; i++) {
            if (flag) {
                System.out.print(map[i]);
                flag = false;
            }
            else {
                System.out.print(" "+map[i]);
            }
        }
    }
}
```

## CCF 201703-4 地铁修建 Java
### 问题描述 
　　A市有n个交通枢纽，其中1号和n号非常重要，为了加强运输能力，A市决定在1号到n号枢纽间修建一条地铁。 
　　地铁由很多段隧道组成，每段隧道连接两个交通枢纽。经过勘探，有m段隧道作为候选，两个交通枢纽之间最多只有一条候选的隧道，没有隧道两端连接着同一个交通枢纽。 
　　现在有n家隧道施工的公司，每段候选的隧道只能由一个公司施工，每家公司施工需要的天数一致。而每家公司最多只能修建一条候选隧道。所有公司同时开始施工。 
　　作为项目负责人，你获得了候选隧道的信息，现在你可以按自己的想法选择一部分隧道进行施工，请问修建整条地铁最少需要多少天。 
输入格式 
　　输入的第一行包含两个整数n, m，用一个空格分隔，分别表示交通枢纽的数量和候选隧道的数量。 
　　第2行到第m+1行，每行包含三个整数a, b, c，表示枢纽a和枢纽b之间可以修建一条隧道，需要的时间为c天。 
输出格式 
　　输出一个整数，修建整条地铁线路最少需要的天数。 
样例输入 
6 6 
1 2 4 
2 3 4 
3 6 7 
1 4 2 
4 5 5 
5 6 6 
样例输出 
6 
样例说明 
　　可以修建的线路有两种。 
　　第一种经过的枢纽依次为1, 2, 3, 6，所需要的时间分别是4, 4, 7，则整条地铁线需要7天修完； 
　　第二种经过的枢纽依次为1, 4, 5, 6，所需要的时间分别是2, 5, 6，则整条地铁线需要6天修完。 
　　第二种方案所用的天数更少。 
评测用例规模与约定 
　　对于20%的评测用例，1 ≤ n ≤ 10，1 ≤ m ≤ 20； 
　　对于40%的评测用例，1 ≤ n ≤ 100，1 ≤ m ≤ 1000； 
　　对于60%的评测用例，1 ≤ n ≤ 1000，1 ≤ m ≤ 10000，1 ≤ c ≤ 1000； 
　　对于80%的评测用例，1 ≤ n ≤ 10000，1 ≤ m ≤ 100000； 
　　对于100%的评测用例，1 ≤ n ≤ 100000，1 ≤ m ≤ 200000，1 ≤ a, b ≤ n，1 ≤ c ≤ 1000000。

　　所有评测用例保证在所有候选隧道都修通时1号枢纽可以通过隧道到达其他所有枢纽。

---
这里采用的是 Kruskal 最小生成树和并查集算法
```
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;

public class Main {
    private int[] flag;

    private int unionFind(int e) {
        while (flag[e] != e) {
            e = flag[e];
        }
        return e;
    }

    private static class Rode implements Comparable<Rode>{
        private int a;
        private int b;
        private int w;
        public Rode(int a, int b, int w) {
            this.a = a;
            this.b = b;
            this.w = w;
        }

        @Override
        public int compareTo(Main.Rode o) {
            if (this.w < o.w)
                return -1;
            else if (this.w > o.w)
                return 1;
            else
                return 0;
        }
    }

    public static void main(String[] args) {
        Scanner In = new Scanner(System.in);
        int N = In.nextInt();
        int m = In.nextInt();
        int ans = 0;
        Main memb = new Main();
        memb.flag = new int[N + 10];
        ArrayList<Rode> roads = new ArrayList<>(200010);
        for (int i = 0; i <= N; i++) {
            memb.flag[i] = i;
        }
        for (int i = 0; i < m; i++) {
            int a = In.nextInt();
            int b = In.nextInt();
            int w = In.nextInt();

            roads.add(new Rode(a, b, w));
        }

        Collections.sort(roads);
        for (int i = 0; i < m; i++) {
            int x = memb.unionFind(roads.get(i).a);
            int y = memb.unionFind(roads.get(i).b);
            if (x != y) {
                if (x > y)
                    memb.flag[x] = y;
                else
                    memb.flag[y] = x;
            }
            ans = roads.get(i).w;

            if (memb.unionFind(N) == 1)
                break;
        }
        System.out.println(ans);
    }
}
```

## CCF 201709-4 通信网络 Java

### 问题描述 
　　某国的军队由N个部门组成，为了提高安全性，部门之间建立了M条通路，每条通路只能单向传递信息，即一条从部门a到部门b的通路只能由a向b传递信息。信息可以通过中转的方式进行传递，即如果a能将信息传递到b，b又能将信息传递到c，则a能将信息传递到c。一条信息可能通过多次中转最终到达目的地。 
　　由于保密工作做得很好，并不是所有部门之间都互相知道彼此的存在。只有当两个部门之间可以直接或间接传递信息时，他们才彼此知道对方的存在。部门之间不会把自己知道哪些部门告诉其他部门。

　　上图中给了一个4个部门的例子，图中的单向边表示通路。部门1可以将消息发送给所有部门，部门4可以接收所有部门的消息，所以部门1和部门4知道所有其他部门的存在。部门2和部门3之间没有任何方式可以发送消息，所以部门2和部门3互相不知道彼此的存在。 
　　现在请问，有多少个部门知道所有N个部门的存在。或者说，有多少个部门所知道的部门数量（包括自己）正好是N。 
输入格式 
　　输入的第一行包含两个整数N, M，分别表示部门的数量和单向通路的数量。所有部门从1到N标号。 
　　接下来M行，每行两个整数a, b，表示部门a到部门b有一条单向通路。 
输出格式 
　　输出一行，包含一个整数，表示答案。 
样例输入 
4 4 
1 2 
1 3 
2 4 
3 4 
样例输出 
2 
样例说明 
　　部门1和部门4知道所有其他部门的存在。 
评测用例规模与约定 
　　对于30%的评测用例，1 ≤ N ≤ 10，1 ≤ M ≤ 20； 
　　对于60%的评测用例，1 ≤ N ≤ 100，1 ≤ M ≤ 1000； 
　　对于100%的评测用例，1 ≤ N ≤ 1000，1 ≤ M ≤ 10000。

```
import java.util.Arrays;
import java.util.Scanner;
import java.util.ArrayList;

public class Main {
    private static int N;
    private static ArrayList<Integer>[] line;
    private static int[][] knows;
    private static int[] visited;
    private static int top;

    private static void dfs(int cur) {
        knows[top][cur] = knows[cur][top] = visited[cur] = 1;
        for (int i = 0; i < line[cur].size(); i++) {
            if (visited[line[cur].get(i)] == 0) {
                dfs(line[cur].get(i));
            }
        }
    }

    public static void main(String[] args) {
        Scanner In = new Scanner(System.in);
        N = In.nextInt();
        int M = In.nextInt();
        int ans = 0;
        line = new ArrayList[N + 1];
        knows = new int[N + 1][N + 1];
        visited = new int[N + 1];
        for (int i = 0; i < line.length; i++) {
            line[i] = new ArrayList<Integer>();
        }

        while (M-- > 0) {
            int l = In.nextInt();
            Integer r = In.nextInt();
            line[l].add(r);
        }

        for (int i = 1; i <= N; i++) {
            Arrays.fill(visited, 0);
            top = i;
            dfs(i);
        }

        for (int i = 1; i <= N; i++) {
            int j = 1;
            for (; j <= N; j++) {
                if (knows[i][j] == 0) {
                    break;
                }
            }
            if (j == N + 1) {
                ans ++;
            }
        }

        System.out.println(ans);
    }
}
```

## CCF 201709-5 除法 Java

### 问题描述 
　　小葱喜欢除法，所以他给了你N个数a1, a2, ⋯, aN，并且希望你执行M次操作，每次操作可能有以下两种： 
　　给你三个数l, r, v，你需要将al, al+1, ⋯, ar之间所有v的倍数除以v。 
　　给你两个数l, r，你需要回答al + al+1 + ⋯ + ar的值是多少。 
输入格式 
　　第一行两个整数N, M，代表数的个数和操作的次数。 
　　接下来一行N个整数，代表N个数一开始的值。 
　　接下来M行，每行代表依次操作。每行开始有一个整数opt。如果opt=1，那么接下来有三个数l, r, v，代表这次操作需要将第l个数到第r个数中v的倍数除以v；如果opt = 2，那么接下来有两个数l, r，代表你需要回答第l个数到第r个数的和。 
输出格式 
　　对于每一次的第二种操作，输出一行代表这次操作所询问的值。 
#### 样例输入 
5 3 
1 2 3 4 5 
2 1 5 
1 1 3 2 
2 1 5 
#### 样例输出 
15 
14 
评测用例规模与约定 
　　对于30%的评测用例，1 ≤ N, M ≤ 1000； 
　　对于另外20%的评测用例，第一种操作中一定有l = r； 
　　对于另外20%的评测用例，第一种操作中一定有l = 1 , r = N； 
　　对于100%的评测用例，1 ≤ N, M ≤ 105，0 ≤ a1, a2, ⋯, aN ≤ 106, 1 ≤ v ≤ 106, 1 ≤ l ≤ r ≤ N。

```java
import java.util.Scanner;

public class Main {

    private static int[] arr = new int[100010];
    private static long[] tarr = new long[100010];
    private static int N;

    private static int lowbit(int i) {
        return i & (-i);
    }

    private static void toValue(int i, int num) {
        while (i <= N) {
            tarr[i] += num;
            i += lowbit(i);
        }
    }

    private static long sum(int i) {
        long total = 0;
        while(i != 0) {
            total += tarr[i];
            i -= lowbit(i);
        }
        return total;
    }

    private static void fun1(int l, int r, int v) {
        if (v == 1) return;
        for (int i = l; i <= r; i++) {
            if (arr[i] >= v && arr[i] % v == 0) {
                toValue(i, arr[i] / v - arr[i]);
                arr[i] /= v;
            }
        }
    }

    private static void fun2(int l, int r) {
        System.out.println(sum(r) - sum(l-1));
    }

    public static void main(String[] args) {
        Scanner In = new Scanner(System.in);
        N = In.nextInt();
        int M = In.nextInt();

        arr = new int[N + 1];
        tarr = new long[N + 1];
        for (int i = 1; i <= N; i++) {
            arr[i] = In.nextInt();
            toValue(i, arr[i]);
        }

        while (M-- > 0) {
            int opt = In.nextInt();
            if (opt == 1) {
                int a = In.nextInt();
                int b = In.nextInt();
                int c = In.nextInt();
                fun1(a, b, c);
            }
            else if (opt == 2) {
                int l = In.nextInt();
                int r = In.nextInt();
                fun2(l, r);
            }
        }
    }
}
```
