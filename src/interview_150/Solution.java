package interview_150;

import java.util.*;

public class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if (m == 0) {                         // need to check edge cases
            if (n >= 0)
                System.arraycopy(nums2, 0, nums1, 0, n);
            return;
        } else if (n == 0) {
            return;
        }
        ArrayList<Integer> elements = new ArrayList<>();
        int lp = 0;
        int rp = 0;
        while (elements.size() < m+n) {       // time complexity: O(m+n)
            if (lp == m) {
                elements.add(nums2[rp]);
                rp++;
            } else if (rp == n) {
                elements.add(nums1[lp]);
                lp++;
            } else {
                int cl = nums1[lp];
                int cr = nums2[rp];
                if (cl <= cr) {
                    elements.add(cl);
                    lp++;
                } else {
                    elements.add(cr);
                    rp++;
                }
            }
        }
        for (int i = 0; i <= elements.size()-1; i++) {
            nums1[i] = elements.get(i);
        }
    }

    public int removeElement(int[] nums, int val) {
        ArrayList<Integer> unique = new ArrayList<>();
        for (int i = 0; i <= nums.length-1; i++) {
            if (nums[i] != val) {
                unique.add(nums[i]);
            }
        }
        for (int j = 0; j <= unique.size()-1; j++) {
            nums[j] = unique.get(j);
        }
        return unique.size();
    }

    public int removeDuplicates_I(int[] nums) {
        ArrayList<Integer> unique = new ArrayList<>();
        for (int i = 0; i <= nums.length-1; i++) {
            if (!unique.contains(nums[i])) {
                unique.add(nums[i]);
            }
        }
        for (int j = 0; j <= unique.size()-1; j++) {
            nums[j] = unique.get(j);
        }
        return unique.size();
    }

    public int removeDuplicates(int[] nums) {
        int re = 0;
        if (nums.length == 1 || nums.length == 2) {
            return nums.length;
        }
        int curr = nums[0];
        int c_index = 0;
        re++;
        for (int i = 1; i <= nums.length-1; i++) {
            if (nums[i] != curr) {
                re++;
                curr = nums[i];
                c_index = i;
            } else {
                if (i == 1) {
                    re++;
                } else {
                    if (i-c_index < 2) {
                        re++;
                    } else {
                        nums[i] = 100000;
                    }
                }
            }
        }
        for (int j = 1; j <= nums.length-2; j++) {
            if (nums[j] == 100000) {
                int flag = 0;
                for (int k = j+1; k <= nums.length-1; k++) {
                    if (nums[k] != 100000) {
                        nums[j] = nums[k];
                        nums[k] = 100000;
                        flag = 1;
                        break;
                    }
                }
                if (flag == 0) {
                    break;
                }
            }
        }
        return re;
    }

    public int majorityElement(int[] nums) {
        if (nums.length <= 2) {
            return nums[0];
        }
        int threshold = nums.length/2;
        int flag = 0;
        int re = nums[0];
        for (int i = 0; i <= nums.length-1; i++) {
            int curr = nums[i];
            int count = 0;
            for (int j = 0; j <= nums.length-1; j++) {
                if (nums[j] == curr) {
                    count++;
                    if (count > threshold) {
                        flag = 1;
                        re = curr;
                        break;
                    }
                }
            }
            if (flag == 1) {
                break;
            }
        }
        return re;
    }

    public void rotate(int[] nums, int k) {
        if (k > 0) {
            k = k % nums.length;
            HashMap<Integer, Integer> map = new HashMap<>();
            for (int i = 0; i <= nums.length - 1; i++) {
                int next = i + k;
                if (next > nums.length - 1) {
                    next = next % nums.length;
                }
                map.put(next, nums[i]);
            }
            for (int j = 0; j <= nums.length - 1; j++) {
                nums[j] = map.get(j);
            }
        }
    }

    public int maxProfit(int[] prices) {
        if (prices.length == 1) {
            return 0;
        }
        int maxProfit = 0;
        int minPrice = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else {
                int profit = prices[i] - minPrice;
                if (profit > maxProfit) {
                    maxProfit = profit;
                }
            }
        }
        return maxProfit;
    }

    public int maxProfit2(int[] prices) {
        if (prices.length == 1) {
            return 0;
        }
        int maxProfit = 0;
        int minPrice = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] <= minPrice) {
                minPrice = prices[i];
            } else {
                int profit = prices[i] - minPrice;
                maxProfit += profit;
                if (i != prices.length-1) {
                    minPrice = prices[i];
                }
            }
        }
        return maxProfit;
    }

    public boolean canJump(int[] nums) {
        int maxReach = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > maxReach) {
                return false;
            }
            maxReach = Math.max(maxReach, i + nums[i]);
            if (maxReach >= nums.length - 1) {
                return true;
            }
        }
        return false;
    }

    public int jump(int[] nums) {
        int count = 0;
        ArrayList<Integer> list = new ArrayList<>();
        list.add(nums.length-1);
        while (!list.isEmpty()) {
            int pos = list.removeFirst();
            for (int i = 0; i <= nums.length-2; i++) {
                int reach = i + nums[i];
                if (reach >= pos) {
                    count++;
                    list.add(i);
                    if (i == 0) {
                        list.clear();
                    }
                    break;
                }
            }
        }
        return count;
    }

    public int hIndex(int[] citations) {
        int index = 0;
        while (true){
            int count = 0;
            for (int i = 0; i <= citations.length-1; i++) {
                if (citations[i] >= index) {
                    count++;
                }
            }
            if (count < index) {
                index--;
                break;
            } else {
                index++;
            }
        }
        return index;
    }

    static class RandomizedSet {
        public LinkedList<Integer> linkedList;
        public HashSet<Integer> hashSet;    // HashSet<> supports constant operation time in add(), remove(), contains()

        public RandomizedSet() {
            linkedList = new LinkedList<>();
            hashSet = new HashSet<>(0);
        }

        public boolean insert(int val) {
            if (hashSet.contains(val)) {
                return false;
            } else {
                linkedList.add(val);
                hashSet.add(val);
                return true;
            }
        }

        public boolean remove(int val) {
            if (hashSet.contains(val)) {
                int index = linkedList.indexOf(val);
                linkedList.set(index, linkedList.getLast());
                linkedList.removeLast();
                hashSet.remove(val);
                return true;
            } else {
                return false;
            }
        }

        public int getRandom() {
            Random random = new Random();
            int index = random.nextInt(linkedList.size());
            return linkedList.get(index);
        }
    }

    public int[] productExceptSelf(int[] nums) {    // Restricted by O(n) time complexity
        int size = nums.length;
        int[] answer = new int[size];
        int[] left = new int[size];
        int[] right = new int[size];
        for (int i = 0; i <= size-1; i++) {
            if (i == 0) {
                left[i] = 1;
            } else {
                left[i] = nums[i-1]*left[i-1];
            }
        }
        for (int j = size-1; j >= 0; j--) {
            if (j == size-1) {
                right[j] = 1;
            } else {
                right[j] = right[j+1]*nums[j+1];
            }
        }
        for (int k = 0; k <= size-1; k++) {
            answer[k] = left[k] * right[k];
        }
        return answer;
    }

    public int canCompleteCircuit(int[] gas, int[] cost) {
        // a more efficient design
        int total = 0;
        int curr = 0;
        int start = 0;
        for (int i = 0; i <= gas.length-1; i++) {
            total += (gas[i]-cost[i]);
            curr += (gas[i]-cost[i]);
            if (curr < 0) {
                start = i+1;
                curr = 0;
            }
        }
        return (total >= 0)?start:-1;
    }

    public int candy(int[] ratings) {
        int n = ratings.length;
        int[] candies = new int[n];
        Arrays.fill(candies, 1);
        if (n == 1) {
            return 1;
        }
        // Left-to-right pass
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }
        // Right-to-left pass
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                candies[i] = Math.max(candies[i], candies[i + 1] + 1);
            }
        }
        // Calculate the total number of candies
        int result = 0;
        for (int candy : candies) {
            result += candy;
        }
        return result;
    }

    public int trap(int[] height) {
        int vol = 0;
        int ml, lp, mr, rp;
        ml = 0;
        lp = -1;
        mr = 0;
        rp = height.length;
        for (int i = 0; i <= height.length-1; i++) {
            if (height[i] != 0) {
                ml = height[i];
                lp = i;
                break;
            }
        }
        for (int j = height.length-1; j >= 0; j--) {
            if (height[j] != 0) {
                mr = height[j];
                rp = j;
                break;
            }
        }
        if ((lp == -1) || (rp == height.length) || (lp == rp)) {
            return 0;
        } else {
            while (lp != rp) {              // "keep track of the maximum heights of left and right"
                if (ml <= mr) {
                    lp++;
                    if (height[lp] > ml) {
                        ml = height[lp];
                    }
                    vol += (ml-height[lp]);
                } else {
                    rp--;
                    if (height[rp] > mr) {
                        mr = height[rp];
                    }
                    vol += (mr-height[rp]);
                }
            }
            return vol;
        }
    }

    public int romanToInt(String s) {           // Basic classification problem
        int pt = 0;
        int result = 0;
        while (pt <= s.length()-1) {
            switch (s.charAt(pt)) {             // 'switch' can be optimized to reduce code length
                case 'I':
                    if (pt != s.length()-1) {
                        if (s.charAt(pt+1) == 'V') {
                            result += 4;
                            pt += 2;
                        } else if (s.charAt(pt+1) == 'X') {
                            result += 9;
                            pt += 2;
                        } else {
                            result += 1;
                            pt += 1;
                        }
                    } else {
                        result += 1;
                        pt += 1;
                    }
                    break;
                case 'V':
                    result += 5;
                    pt += 1;
                    break;
                case 'X':
                    if (pt != s.length()-1) {
                        if (s.charAt(pt+1) == 'L') {
                            result += 40;
                            pt += 2;
                        } else if (s.charAt(pt+1) == 'C') {
                            result += 90;
                            pt += 2;
                        } else {
                            result += 10;
                            pt += 1;
                        }
                    } else {
                        result += 10;
                        pt += 1;
                    }
                    break;
                case 'L':
                    result += 50;
                    pt += 1;
                    break;
                case 'C':
                    if (pt != s.length()-1) {
                        if (s.charAt(pt+1) == 'D') {
                            result += 400;
                            pt += 2;
                        } else if (s.charAt(pt+1) == 'M') {
                            result += 900;
                            pt += 2;
                        } else {
                            result += 100;
                            pt += 1;
                        }
                    } else {
                        result += 100;
                        pt += 1;
                    }
                    break;
                case 'D':
                    result += 500;
                    pt += 1;
                    break;
                case 'M':
                    result += 1000;
                    pt += 1;
                    break;
            }
        }
        return result;
    }

    public String intToRoman(int num) {
        StringBuilder stringBuffer = new StringBuilder();
        int pt;
        int d4 = num/1000;
        pt = d4;
        while (pt >= 1) {
            stringBuffer.append('M');
            pt--;
        }
        int d3 = (num-d4*1000)/100;
        pt = d3;
        if (d3 >= 5) {
            if (d3 == 9) {
                stringBuffer.append("CM");
            } else {
                stringBuffer.append('D');
                pt -= 5;
                while (pt >= 1) {
                    stringBuffer.append('C');
                    pt--;
                }
            }
        } else {
            if (d3 == 4) {
                stringBuffer.append("CD");
            } else {
                while (pt >= 1) {
                    stringBuffer.append('C');
                    pt--;
                }
            }
        }
        int d2 = (num-d4*1000-d3*100)/10;
        pt = d2;
        if (d2 >= 5) {
            if (d2 == 9) {
                stringBuffer.append("XC");
            } else {
                stringBuffer.append('L');
                pt -= 5;
                while (pt >= 1) {
                    stringBuffer.append('X');
                    pt--;
                }
            }
        } else {
            if (d2 == 4) {
                stringBuffer.append("XL");
            } else {
                while (pt >= 1) {
                    stringBuffer.append('X');
                    pt--;
                }
            }
        }
        int d1 = num-d4*1000-d3*100-d2*10;
        pt = d1;
        if (d1 >= 5) {
            if (d1 == 9) {
                stringBuffer.append("IX");
            } else {
                stringBuffer.append('V');
                pt -= 5;
                while (pt >= 1) {
                    stringBuffer.append('I');
                    pt--;
                }
            }
        } else {
            if (d1 == 4) {
                stringBuffer.append("IV");
            } else {
                while (pt >= 1) {
                    stringBuffer.append('I');
                    pt--;
                }
            }
        }
        return stringBuffer.toString();
    }

    public int lengthOfLastWord(String s) {
        String[] list = s.split(" ");       // regex is String
        return list[list.length-1].length();
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs.length == 1) {
            return strs[0];
        }
        StringBuilder stringBuffer = new StringBuilder();    // StringBuffer is thread-safe, while StringBuilder is not.
                                                             // However, StringBuilder is more efficient.
        String shortest = strs[0];
        for (String s: strs) {
            if (s.length() < shortest.length()) {
                shortest = s;
            }
        }
        for (int i = 0; i <= shortest.length()-1; i++) {
            char tar = shortest.charAt(i);
            int flag = 1;
            for (String t: strs) {
                if (t.charAt(i) != tar) {
                    flag = 0;
                    break;
                }
            }
            if (flag == 1) {
                stringBuffer.append(tar);
            } else {
                return stringBuffer.toString();
            }
        }
        return stringBuffer.toString();
    }

    public String reverseWords(String s) {
        StringBuilder stringBuffer = new StringBuilder();
        String[] strs = s.split(" +");      // Use + for matching patterns with multiple spaces
        for (int i = strs.length-1; i >= 0; i--) {
            stringBuffer.append(strs[i]);
            if (i != 0) {
                stringBuffer.append(" ");
            }
        }
        if (stringBuffer.charAt(stringBuffer.length()-1) == ' ') {   // if the original String has space(s) at the start
            stringBuffer.setLength(stringBuffer.length()-1);   // cut off the last character
        }
        return stringBuffer.toString();
    }

    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s;
        }
        StringBuilder stringBuffer = new StringBuilder();
        for (int i = numRows; i >= 1; i--) {
            int gap = (numRows-1)*2;
            int jump;
            if (i != 1) {
                jump = (i-1)*2;
            } else {
                jump = (numRows-1)*2;
            }
            int j = numRows-i;
            while (j <= s.length()-1) {
                stringBuffer.append(s.charAt(j));
                j += jump;
                if (i != numRows && i != 1) {     // alternating pattern between the corners
                    jump = gap-jump;
                }
            }
        }
        return stringBuffer.toString();
    }

    public int strStr(String haystack, String needle) {
        if (!haystack.contains(needle)) {
            return -1;
        } else {
            return haystack.indexOf(needle);   // indexOf() can also have a String as input
        }
    }

    public List<String> fullJustify(String[] words, int maxWidth) {
        // General idea: first find out which words belong to each row,
        //               then find out how spaces are distributed between words in each row and build the string.
        List<String> list = new ArrayList<>();
        if (words.length == 1) {
            StringBuilder stringBuffer = new StringBuilder(words[0]);
            int pad = maxWidth-words[0].length();
            while (pad > 0) {
                stringBuffer.append(' ');
                pad--;
            }
            list.add(stringBuffer.toString());
        } else {
            // Find out the mapping from rows to words.
            Map<Integer, List<String>> map = new HashMap<>();
            int row = 0;
            map.put(row, new ArrayList<>());
            for (String s: words) {
                if (map.get(row).isEmpty()) {
                    map.get(row).add(s);
                } else {
                    int accu = 0;
                    for (String s1: map.get(row)) {
                        accu += s1.length();
                        accu += 1;
                    }
                    if (maxWidth-accu >= s.length()) {
                        map.get(row).add(s);
                    } else {
                        row += 1;
                        map.put(row, new ArrayList<>());
                        map.get(row).add(s);
                    }
                }
            }
            for (int i = 0; i <= row; i++) {
                StringBuilder stringBuffer = new StringBuilder();
                if (i != row) {
                    if (map.get(i).size() != 1) {
                        // Find out how spaces are distributed.
                        int spaces = map.get(i).size()-1;
                        int whites = maxWidth;
                        for (String s: map.get(i)) {
                            whites -= s.length();
                        }
                        int[] counts = new int[spaces];  // default initialization to 0 for int[]
                        int iti = 0;
                        while (whites > 0) {
                            counts[iti%spaces] += 1;
                            whites -= 1;
                            iti += 1;
                        }
                        // Build the string.
                        for (int k = 0; k <= map.get(i).size()-1; k++) {
                            stringBuffer.append(map.get(i).get(k));
                            if (k != map.get(i).size()-1) {
                                int c = counts[k];
                                while (c > 0) {
                                    stringBuffer.append(' ');
                                    c--;
                                }
                            }
                        }
                    } else {
                        stringBuffer.append(map.get(i).getFirst());
                        while (stringBuffer.length() < maxWidth) {
                            stringBuffer.append(' ');
                        }
                    }
                } else {
                    for (int j = 0; j <= map.get(i).size()-1; j++) {
                        stringBuffer.append(map.get(i).get(j));
                        if (j != map.get(i).size()-1) {
                            stringBuffer.append(' ');
                        }
                    }
                    while (stringBuffer.length() < maxWidth) {
                        stringBuffer.append(' ');
                    }
                }
                list.add(stringBuffer.toString());
            }
        }
        return list;
    }

    public boolean isPalindrome(String s) {           // demonstration of the Character class
        StringBuilder stringBuffer = new StringBuilder();
        for (int i = 0; i <= s.length()-1; i++) {
            char c = s.charAt(i);
            if (Character.isAlphabetic(c) || Character.isDigit(c)) {
                if (Character.isAlphabetic(c)) {
                    if (Character.isUpperCase(c)) {
                        stringBuffer.append(Character.toLowerCase(c));
                    } else {
                        stringBuffer.append(c);
                    }
                } else {
                    stringBuffer.append(c);
                }
            }
        }
        if (stringBuffer.isEmpty() || stringBuffer.length() == 1) {
            return true;
        }
        for (int j = 0; j <= stringBuffer.length()/2-1; j++) {
            if (stringBuffer.charAt(j) != stringBuffer.charAt(stringBuffer.length()-1-j)) {
                return false;
            }
        }
        return true;
    }

    public boolean isSubsequence(String s, String t) {     // time complexity: O(|s|*|t|)
        if (s.isEmpty()) {
            return true;
        }
        if (t.length() < s.length()) {
            return false;
        }
        int st = 0;
        for (int i = 0; i <= s.length()-1; i++) {
            if (t.substring(st).indexOf(s.charAt(i)) != -1) {  // repetitive substring() is inefficient
                st = t.substring(st).indexOf(s.charAt(i))+st+1;
            } else {
                return false;
            }
        }
        return true;
    }

    public int[] twoSum(int[] numbers, int target) {
        // A more efficient approach
        // Use 2 pointers, moving towards each other.
        // Time complexity: O(N)
        int[] answer = new int[2];
        int lp = 0;
        int rp = numbers.length-1;
        while (numbers[lp]+numbers[rp] != target) {
            if (numbers[lp]+numbers[rp] > target) {
                rp-=1;
            } else {
                lp+=1;
            }
        }
        answer[0] = lp+1;
        answer[1] = rp+1;
        return answer;
    }

    public int maxArea(int[] height) {
        int lp = 0;
        int rp = height.length-1;
        int area = 0;
        while (lp <= rp-1) {
            int low = (Math.min(height[lp], height[rp]));
            int curr = low*(rp-lp);
            area = (Math.max(curr, area));
            if (lp == rp-1) {
                break;
            }
            if (height[lp] <= height[rp]) {
                lp += 1;
            } else {
                rp -= 1;
            }
        }
        return area;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> container = new ArrayList<>(0);
        Set<String> unique = new HashSet<>(0);
        Arrays.sort(nums);
        ArrayList<Integer> third = new ArrayList<>();
        for (int i = 0; i <= nums.length-1; i++) {
            if (!third.contains(nums[i])) {
                third.add(nums[i]);
                int lp = (i==0? 1: 0);
                int rp = (i==nums.length-1? nums.length-2: nums.length-1);
                int lv;
                while (lp < rp) {
                    if (nums[lp]+nums[rp]+nums[i] == 0) {
                        List<Integer> list = new ArrayList<>();
                        list.add(nums[i]);
                        list.add(nums[lp]);
                        list.add(nums[rp]);
                        Collections.sort(list);
                        String contents = list.toString();
                        if (!unique.contains(contents)) {
                            container.add(list);
                            unique.add(contents);
                        }
                        lv = nums[lp];
                        do {
                            lp++;
                        } while ((nums[lp] == lv || lp == i) && lp < rp);
                    } else {
                        if (nums[lp]+nums[rp]+nums[i] > 0) {
                            rp--;
                            if (rp == i) {
                                rp--;
                            }
                        } else {
                            lp++;
                            if (lp == i) {
                                lp++;
                            }
                        }
                    }
                }
            }
        }
        return container;
    }

    public int minSubArrayLen(int target, int[] nums) {
        // Idea: maintain 2 pointers, right and left, both starting at 0;
        //       increment the right pointer, when the target is met, update the length;
        //       then shrink the length by incrementing the left pointer until target is missed again.
        //   Time complexity: O(N)
        int n = nums.length;
        int minLen = Integer.MAX_VALUE;
        int left = 0;
        int sum = 0;
        for (int right = 0; right < n; right++) {
            sum += nums[right];

            while (sum >= target) {
                minLen = Math.min(minLen, right - left + 1);
                sum -= nums[left];
                left++;
            }
        }
        return minLen == Integer.MAX_VALUE ? 0 : minLen;
    }

    public int lengthOfLongestSubstring(String s) {
        if (s.isEmpty()) {
            return 0;
        } else if (s.length() == 1) {
            return 1;
        } else {
            int length = 0;
            int start = 0;
            while (start <= s.length()-1) {
                StringBuilder stringBuffer = new StringBuilder();
                stringBuffer.append(s.charAt(start));
                int right = start+1;
                while (right <= s.length()-1) {
                    if (stringBuffer.toString().indexOf(s.charAt(right)) == -1) {
                        stringBuffer.append(s.charAt(right));
                        right++;
                    } else {
                        break;
                    }
                }
                if (stringBuffer.length() > length) {
                    length = stringBuffer.length();
                }
                start++;
            }
            return length;
        }
    }

    public List<Integer> findSubstring(String s, String[] words) {
        // Solution provided by LeetCode.
        List<Integer> ans = new ArrayList<>();
        int n = s.length();
        int m = words.length;
        int w = words[0].length();
        HashMap<String, Integer> map = new HashMap<>();
        for(String x: words)
            map.put(x, map.getOrDefault(x, 0)+1);
        for(int i=0; i<w; i++){
            HashMap<String, Integer> temp = new HashMap<>();
            int count = 0;
            for(int j=i, k=i; j+w<=n; j=j+w){
                String word = s.substring(j, j+w);
                temp.put(word, temp.getOrDefault(word, 0)+1);
                count++;
                if(count==m){
                    if(map.equals(temp)){
                        ans.add(k);
                    }
                    String remove = s.substring(k, k+w);
                    temp.computeIfPresent(remove, (a, b) -> (b > 1) ? b - 1 : null);
                    count--;
                    k=k+w;
                }
            }
        }
        return ans;
    }

    public String minWindow(String s, String t) {
        // A more efficient solution.
        HashMap<Character, Integer> map = new HashMap<>();
        for (char c : t.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        int left = 0, right = 0, minLen = Integer.MAX_VALUE, minStart = 0, count = 0;
        while (right < s.length()) {
            char c = s.charAt(right);
            if (map.containsKey(c)) {
                map.put(c, map.get(c) - 1);
                if (map.get(c) >= 0) {
                    count++;
                }
            }
            right++;
            while (count == t.length()) {
                if (right - left < minLen) {
                    minLen = right - left;
                    minStart = left;
                }
                char c2 = s.charAt(left);
                if (map.containsKey(c2)) {
                    map.put(c2, map.get(c2) + 1);
                    if (map.get(c2) > 0) {
                        count--;
                    }
                }
                left++;
            }
        }
        return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
    }

    public boolean isValidSudoku(char[][] board) {
        ArrayList<Character> arrayList = new ArrayList<>();
        for (int i = 0; i <= board.length-1; i++) {
            for (int j = 0; j <= board[0].length-1; j++) {
                if (board[i][j] != '.') {
                    if (!arrayList.contains(board[i][j])) {
                        arrayList.add(board[i][j]);
                    } else {
                        return false;
                    }
                }
            }
            arrayList.clear();
        }
        for (int m = 0; m <= board[0].length-1; m++) {
            for (int n = 0; n <= board.length-1; n++) {
                if (board[n][m] != '.') {
                    if (!arrayList.contains(board[n][m])) {
                        arrayList.add(board[n][m]);
                    } else {
                        return false;
                    }
                }
            }
            arrayList.clear();
        }
        for (int o = 0; o <= 8; o++) {
            int r = o/3;
            int c = o%3;
            for (int p = r*3; p <= r*3+2; p++) {
                for (int q = c*3; q <= c*3+2; q++) {
                    if (board[p][q] != '.') {
                        if (!arrayList.contains(board[p][q])) {
                            arrayList.add(board[p][q]);
                        } else {
                            return false;
                        }
                    }
                }
            }
            arrayList.clear();
        }
        return true;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        int rows = matrix.length;
        int columns = matrix[0].length;
        int total = rows*columns;
        List<Integer> list = new ArrayList<>();
        boolean horizontal = true;
        boolean vertical = true;
        int r = 0;
        int c = 0;
        ArrayList<Integer> rt = new ArrayList<>();
        ArrayList<Integer> ct = new ArrayList<>();
        while (total > 0) {
            if (horizontal && vertical) {
                while (!ct.contains(c) && c <= columns-1) {
                    list.add(matrix[r][c]);
                    total--;
                    c++;
                }
                c--;
                rt.add(r);
                r++;
                horizontal = false;
            } else if (!horizontal && vertical) {
                while (!rt.contains(r) && r <= rows-1) {
                    list.add(matrix[r][c]);
                    total--;
                    r++;
                }
                r--;
                ct.add(c);
                c--;
                vertical = false;
            } else if (!horizontal) {
                while (!ct.contains(c) && c >= 0) {
                    list.add(matrix[r][c]);
                    total--;
                    c--;
                }
                c++;
                rt.add(r);
                r--;
                horizontal = true;
            } else {
                while (!rt.contains(r) && r >= 0) {
                    list.add(matrix[r][c]);
                    total--;
                    r--;
                }
                r++;
                ct.add(c);
                c++;
                vertical = true;
            }
        }
        return list;
    }

    public void rotate(int[][] matrix) {
        if (matrix.length != 1) {
            int mid = (matrix.length-1)/2;
            for (int i = 0; i <= mid; i++) {
                for (int j = i; j <= matrix[0].length-2-i; j++) {
                    int step = 1;
                    int curr_row = i;
                    int curr_col = j;
                    int replacer = matrix[i][j];
                    while (step <= 4) {
                        int new_col = matrix.length-1-curr_row;
                        int new_row = curr_col;
                        int temp = matrix[new_row][new_col];
                        matrix[new_row][new_col] = replacer;
                        replacer = temp;
                        curr_row = new_row;
                        curr_col = new_col;
                        step++;
                    }
                }
            }
        }
    }

    public void setZeroes(int[][] matrix) {
        if (matrix.length != 1 || matrix[0].length != 1) {
            ArrayList<Integer> rows = new ArrayList<>();
            ArrayList<Integer> cols = new ArrayList<>();
            for (int i = 0; i <= matrix.length-1; i++) {
                for (int j = 0; j <= matrix[0].length-1; j++) {
                    if (matrix[i][j] == 0) {
                        rows.add(i);
                        cols.add(j);
                    }
                }
            }
            for (int p = 0; p <= matrix.length-1; p++) {
                if (rows.contains(p)) {
                    for (int p1 = 0; p1 <= matrix[0].length-1; p1++) {
                        matrix[p][p1] = 0;
                    }
                } else {
                    for (int q = 0; q <= matrix[0].length-1; q++) {
                        if (cols.contains(q)) {
                            matrix[p][q] = 0;
                        }
                    }
                }
            }
        }
    }

    public void gameOfLife(int[][] board) {
        if (board.length == 1 && board[0].length == 1) {
            board[0][0] = 0;
        } else {
            ArrayList<Integer> next = new ArrayList<>();
            for (int i = 0; i <= board.length-1; i++) {
                for (int j = 0; j <= board[0].length-1; j++) {
                    int live = 0;
                    for (int m = i-1; m <= i+1; m++) {
                        if (m < 0 || m > board.length-1)
                            continue;
                        for (int n = j-1; n <= j+1; n++) {
                            if (n < 0 || n > board[0].length-1 || (m == i && n == j))
                                continue;
                            if (board[m][n] == 1)
                                live++;
                        }
                    }
                    if (board[i][j] == 1) {
                        if (live < 2 || live > 3)
                            next.add(0);
                        else
                            next.add(1);
                    } else {
                        if (live == 3)
                            next.add(1);
                        else
                            next.add(0);
                    }
                }
            }
            int pt = 0;
            for (int i = 0; i <= board.length-1; i++) {
                for (int j = 0; j <= board[0].length-1; j++) {
                    board[i][j] = next.get(pt);
                    pt++;
                }
            }
        }
    }

    public boolean canConstruct(String ransomNote, String magazine) {
        if (ransomNote.length() > magazine.length()) {
            return false;
        }
        HashMap<Character, Integer> ran = new HashMap<>();
        for (int i = 0; i <= ransomNote.length()-1; i++) {
            char curr = ransomNote.charAt(i);
            ran.put(curr, ran.getOrDefault(curr, 0)+1);
        }
        HashMap<Character, Integer> maga = new HashMap<>();
        for (int i = 0; i <= magazine.length()-1; i++) {
            char curr = magazine.charAt(i);
            maga.put(curr, maga.getOrDefault(curr, 0)+1);
        }
        for (Character key: ran.keySet()) {
            if (!maga.containsKey(key)) {
                return false;
            }
            if (ran.get(key) > maga.get(key)) {
                return false;
            }
        }
        return true;
    }

    public boolean isIsomorphic(String s, String t) {
        HashSet<Character> ss = new HashSet<>();     // HashSet<>: every item inside is unique
        HashSet<Character> ts = new HashSet<>();
        HashMap<Character, Character> stt = new HashMap<>();
        for (int i = 0; i <= s.length()-1; i++) {
            char cs = s.charAt(i);
            char ct = t.charAt(i);
            if ((!ss.contains(cs)&&ts.contains(ct)) || (ss.contains(cs)&&!ts.contains(ct))) {
                return false;
            }
            if (!ss.contains(cs)&&!ts.contains(ct)) {
                ss.add(cs);
                ts.add(ct);
                stt.put(cs, ct);
            } else {
                char exp = stt.get(cs);
                if (exp != ct) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean wordPattern(String pattern, String s) {
        String[] words = s.split(" ");
        if (pattern.length() != words.length) {
            return false;
        }
        HashMap<Character, String> ps = new HashMap<>();
        for (int i = 0; i <= pattern.length()-1; i++) {
            char cc = pattern.charAt(i);
            String cs = words[i];
            if (!ps.containsKey(cc)&&ps.containsValue(cs) || ps.containsKey(cc)&&!ps.containsValue(cs)) {
                return false;
            } else if (ps.containsKey(cc)&&ps.containsValue(cs)) {
                String exp = ps.get(cc);
                if (!cs.equals(exp)) {
                    return false;
                }
            } else {
                ps.put(cc, cs);
            }
        }
        return true;
    }

    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        HashMap<Character, Integer> sm = new HashMap<>();
        HashMap<Character, Integer> tm = new HashMap<>();
        for (int i = 0; i <= s.length()-1; i++) {
            sm.put(s.charAt(i), sm.getOrDefault(s.charAt(i), 0)+1);
            tm.put(t.charAt(i), tm.getOrDefault(t.charAt(i), 0)+1);
        }
        for (Character key: sm.keySet()) {
            if (!tm.containsKey(key)) {
                return false;
            } else {
                if (!sm.get(key).equals(tm.get(key))) {
                    return false;
                }
            }
        }
        return true;
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        // A more efficient approach.
        // Time complexity: O(n*k*log(k))
        List<List<String>> list = new ArrayList<>();
        if (strs.length == 1) {
            List<String> self = new ArrayList<>();
            self.add(strs[0]);
            list.add(self);
            return list;
        }
        HashMap<String, List<String>> map = new HashMap<>();
        for (int i = 0; i <= strs.length-1; i++) {
            char[] pattern = strs[i].toCharArray();
            Arrays.sort(pattern);          // Arrays.sort() uses 2-pivot quicksort, time complexity: O(n*log(n))
            String ana = new String(pattern);
            if (!map.containsKey(ana)) {
                map.put(ana, new ArrayList<>());
            }
            map.get(ana).add(strs[i]);
        }
        list.addAll(map.values());
        return list;
    }
}
