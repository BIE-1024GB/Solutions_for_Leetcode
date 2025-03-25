package interview_150;

import java.util.*;

/**
 * @author Jiarui BIE
 * @version 1.1
 * @since 2024/6/24
 */
public class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        if (m == 0) {                         // need to check edge cases
            if (n >= 0) {
                System.arraycopy(nums2, 0, nums1, 0, n);
            }
            return;
        } else if (n == 0) {
            return;
        }
        int p1 = m-1;
        int p2 = n-1;
        int pm = m+n-1;
        while (p1 >= 0 && p2 >= 0) {
            if (nums1[p1] > nums2[p2]) {
                nums1[pm] = nums1[p1];
                p1--;
            } else {
                nums1[pm] = nums2[p2];
                p2--;
            }
            pm--;
        }
        while (p2 >= 0) {
            nums1[pm] = nums2[p2];
            p2--;
            pm--;
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

    public int[] twoSum_II(int[] numbers, int target) {
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

    public int[] twoSum(int[] nums, int target) {
        // O(n) time and space complexity using HashMap<>.
        int[] result = new int[2];
        HashMap<Integer, Integer> add = new HashMap<>();
        HashMap<Integer, Integer> rev = new HashMap<>();
        for (int i = 0; i <= nums.length-1; i++) {
            if (add.containsValue(nums[i])) {
                result[0] = rev.get(nums[i]);
                result[1] = i;
                break;
            }
            int residue = target-nums[i];
            add.put(i, residue);
            rev.put(residue, i);
        }
        return result;
    }

    public boolean isHappy(int n) {
        // Time complexity: O(log(n)), space complexity: O(1)
        HashSet<Integer> hashSet = new HashSet<>();
        // This while loop will be bounded by a CONSTANT time, thus O(1) complexity
        // Explanation: for 1<=n<=2^31-1(2147483647), the maximum possible sum of squares of digits
        // will come from: 1999999999, the sum is 730.
        // Therefore, the number of loops, and space for the HashSet<>, are bounded by this constant integer.
        while (n != 1 && !hashSet.contains(n)) {
            hashSet.add(n);
            int sum = 0;
            // This while loop will have O(log(n)) time to separate each digit
            while (n > 0) {
                int dig = n%10;
                sum += dig*dig;
                n /= 10;
            }
            n = sum;
        }
        return n == 1;
    }

    public boolean containsNearbyDuplicate(int[] nums, int k) {
        if (nums.length == 1 || k == 0) {
            return false;
        }
        HashMap<Integer, ArrayList<Integer>> map = new HashMap<>();
        for (int i = 0; i <= nums.length-1; i++) {
            if (!map.containsKey(nums[i])) {
                map.put(nums[i], new ArrayList<>());
                map.get(nums[i]).add(i);
            } else {
                int near = map.get(nums[i]).getLast();
                if (i-near <= k) {
                    return true;
                } else {
                    map.get(nums[i]).add(i);
                }
            }
        }
        return false;
    }

    public int longestConsecutive(int[] nums) {
        if (nums.length <= 1) {
            return nums.length;
        }
        HashSet<Integer> numSet = new HashSet<>();
        for (int num : nums) {
            numSet.add(num);
        }
        int longestStreak = 0;
        for (int num : numSet) {
            // Check if num is the start of a sequence
            if (!numSet.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;
                while (numSet.contains(currentNum + 1)) {
                    currentNum++;
                    currentStreak++;
                }
                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }
        return longestStreak;
    }

    public List<String> summaryRanges(int[] nums) {
        List<String> list = new ArrayList<>();
        if (nums.length == 0) {
            return list;
        }
        for (int i = 0; i <= nums.length-1; i++) {
            int curr = nums[i];
            if (i == nums.length-1) {
                list.add(String.valueOf(nums[i]));
            } else {
                while (i <= nums.length-2 && nums[i+1]-nums[i] == 1) {
                    i++;
                }
                if (nums[i] != curr) {
                    String pre = String.valueOf(curr);
                    String post = String.valueOf(nums[i]);
                    list.add(pre.concat("->").concat(post));
                } else {
                    list.add(String.valueOf(curr));
                }
            }
        }
        return list;
    }

    public int[][] merge(int[][] intervals) {
        // Sort intervals by the start time
        Arrays.sort(intervals, Comparator.comparingInt(a -> a[0]));
        LinkedList<int[]> merged = new LinkedList<>();
        for (int[] interval : intervals) {
            // If the list of merged intervals is empty or if the current interval does not overlap with the previous,
            // simply append it.
            if (merged.isEmpty() || merged.getLast()[1] < interval[0]) {
                merged.add(interval);
            } else {
                // There is an overlap, so merge the current and previous intervals.
                merged.getLast()[1] = Math.max(merged.getLast()[1], interval[1]);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }

    public int[][] insert(int[][] intervals, int[] newInterval) {
        LinkedList<int[]> merged = new LinkedList<>();
        if (intervals.length == 0) {
            merged.add(newInterval);
        } else {
            // Use another List<> to avoid creating arrays.
            LinkedList<int[]> full = new LinkedList<>(Arrays.asList(intervals));
            full.add(newInterval);
            full.sort(Comparator.comparingInt(value -> value[0]));
            for (int[] f: full) {
                if (merged.isEmpty() || merged.getLast()[1] < f[0]) {
                    merged.add(f);
                } else {
                    merged.getLast()[1] = Math.max(merged.getLast()[1], f[1]);
                }
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }

    public int findMinArrowShots(int[][] points) {
        // Greedy approach: first sort the array by the ending points, then iterate;
        //                  keep track of the "end" variable: update when a new arrow is needed(no overlapping).
        if (points.length == 1)
            return 1;
        Arrays.sort(points, Comparator.comparingInt(a -> a[1]));
        int arrows = 1;
        int end = points[0][1];
        for (int i = 1; i <= points.length-1; i++) {
            if (points[i][0] > end) {
                arrows++;
                end = points[i][1];
            }
        }
        return arrows;
    }

    public boolean isValid(String s) {
        if (s.length()%2 == 1)
            return false;
        Stack<Character> st = new Stack<>();    // explicit Stack<> class in Java
        for (int i = 0; i <= s.length()-1; i++) {
            char curr = s.charAt(i);
            if (curr == '(' || curr == '{' || curr == '[') {
                st.push(curr);
            } else {
                if (st.isEmpty()) {
                    return false;
                } else {
                    char pp = st.pop();
                    if (curr == ')') {
                        if (pp != '(')
                            return false;
                    } else if (curr == '}') {
                        if (pp != '{')
                            return false;
                    } else {
                        if (pp != '[')
                            return false;
                    }
                }
            }
        }
        return st.isEmpty();
    }

    public String simplifyPath(String path) {
        if (path.length() == 1) {
            return "/";
        }
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append('/');
        LinkedList<String> list = new LinkedList<>();
        String[] contexts = path.split("/+");
        if (contexts.length >= 1) {
            for (int i = 0; i <= contexts.length-1; i++) {
                String curr = contexts[i];
                if (!curr.isEmpty() && !curr.equals(".")) {
                    if (curr.equals("..")) {
                        if (!list.isEmpty()) {
                            list.removeLast();
                        }
                    } else {
                        list.add(curr);
                    }
                }
            }
            for (int j = 0; j <= list.size()-1; j++) {
                stringBuilder.append(list.get(j));
                if (j != list.size()-1) {
                    stringBuilder.append('/');
                }
            }
        }
        return stringBuilder.toString();
    }

    static class MinStack {
        // Requirement: O(1) for all methods.
        private final Stack<Integer> st;
        private final Stack<Integer> mst;

        public MinStack() {
            st = new Stack<>();
            mst = new Stack<>();
        }

        public void push(int val) {
            st.push(val);
            if (mst.isEmpty() || val <= mst.peek()) {
                mst.push(val);
            }
        }

        public void pop() {
            if (st.peek().equals(mst.peek())) {
                mst.pop();
            }
            st.pop();
        }

        public int top() {
            return st.peek();
        }

        public int getMin() {
            return mst.peek();
        }
    }

    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (String str: tokens) {
            if (str.equals("+") || str.equals("-") || str.equals("*") || str.equals("/")) {
                int op2 = stack.pop();
                int op1 = stack.pop();
                switch (str) {
                    case "+" -> stack.push(op1 + op2);
                    case "-" -> stack.push(op1 - op2);
                    case "*" -> stack.push(op1 * op2);
                    default -> stack.push(op1 / op2);
                }
            } else {
                stack.push(Integer.parseInt(str));
            }
        }
        return stack.peek();
    }

    public int calculate(String s) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i <= s.length()-1; i++) {
            char curr = s.charAt(i);
            if (curr != ' ') {
                // A variation: unary '-' is permitted
                if (curr == '-') {
                    if (stringBuilder.isEmpty() || stringBuilder.charAt(stringBuilder.length()-1) == '(') {
                        stringBuilder.append('0');
                    }
                }
                stringBuilder.append(curr);
            }
        }
        Stack<Character> ops = new Stack<>();    // a stack for (, ), +, -, *, /
        ArrayList<String> out = new ArrayList<>();
        // The shunting yard algorithm by Dijkstra
        for (int i = 0; i <= stringBuilder.length()-1; i++) {
            char token = stringBuilder.charAt(i);
            // Add the number to the output directly
            if (Character.isDigit(token)) {
                StringBuilder number = new StringBuilder();
                while (i <= stringBuilder.length()-1 && Character.isDigit(stringBuilder.charAt(i))) {
                    number.append(stringBuilder.charAt(i));
                    i++;
                }
                i--;
                out.add(number.toString());
            } else if (token == '(') {
                // Add '(' to the stack directly
                ops.push(token);
            } else if (token == ')') {
                // Add the contents within a "()"
                // '(' and ')' won't appear in the output
                while (ops.peek() != '(') {
                    out.add(String.valueOf(ops.pop()));
                }
                ops.pop();
            } else {
                // Handle operators
                // Precedence: "*, /" > "+, -" > "(, )"
                // Keep popping elements from the stack to the output,
                // until the top element's precedence is LOWER than the current,
                // push the current operator at last.
                int pre = switch (token) {
                    case '+', '-' -> 1;
                    default -> 2;
                };
                while (!ops.isEmpty()) {
                    int top = switch (ops.peek()) {
                        case '(', ')' -> 0;
                        case '+', '-' -> 1;
                        default -> 2;
                    };
                    if (top >= pre) {
                        out.add(String.valueOf(ops.pop()));
                    } else {
                        break;
                    }
                }
                ops.push(token);
            }
        }
        while (!ops.isEmpty()) {
            out.add(String.valueOf(ops.pop()));
        }
        // Conversion complete, now start evaluation
        Stack<Integer> stack = new Stack<>();
        for (String str: out) {
            if (str.equals("+") || str.equals("-") || str.equals("*") || str.equals("/")) {
                int op2 = stack.pop();
                int op1 = stack.pop();
                switch (str) {
                    case "+" -> stack.push(op1 + op2);
                    case "-" -> stack.push(op1 - op2);
                    case "*" -> stack.push(op1 * op2);
                    default -> stack.push(op1 / op2);
                }
            } else {
                stack.push(Integer.parseInt(str));
            }
        }
        return stack.peek();
    }

    static class ListNode {
        int val;
        ListNode next;

        ListNode() {

        }
        ListNode(int x) {
            val = x;
            next = null;
        }
        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }

        public boolean hasCycle(ListNode head) {
            if (head == null || head.next == null) {
                return false;
            }
            HashSet<ListNode> set = new HashSet<>();
            ListNode curr = head;
            while (curr.next != null) {
                if (set.contains(curr)) {
                    return true;
                } else {
                    set.add(curr);
                    curr = curr.next;
                }
            }
            return false;
        }

        public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
            ListNode head = (l1.val+l2.val>=10?new ListNode(l1.val+l2.val-10):new ListNode(l1.val+l2.val));
            int carry = (l1.val+l2.val>=10?1:0);
            ListNode pre = head;
            while (l1.next != null && l2.next != null) {
                l1 = l1.next;
                l2 = l2.next;
                ListNode nx = (l1.val+l2.val+carry>=10?new ListNode(l1.val+l2.val+carry-10):new ListNode(l1.val+l2.val+carry));
                pre.next = nx;
                carry = (l1.val+l2.val+carry>=10?1:0);
                pre = nx;
            }
            if (l1.next != null || l2.next != null) {
                ListNode sl = (l1.next == null?l2:l1);
                while (sl.next != null) {
                    sl = sl.next;
                    ListNode nx = (sl.val+carry>=10?new ListNode(sl.val+carry-10):new ListNode(sl.val+carry));
                    pre.next = nx;
                    carry = (sl.val+carry>=10?1:0);
                    pre = nx;
                }
            }
            if (carry == 1) {
                pre.next = new ListNode(1);
            }
            return head;
        }

        public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
            if (list1 == null && list2 == null) {
                return null;
            } else if (list1 == null) {
                return list2;
            } else if (list2 == null) {
                return list1;
            } else {
                ListNode head, curr;
                if (list1.val > list2.val) {
                    head = list2;
                    list2 = list2.next;
                } else {
                    head = list1;
                    list1 = list1.next;
                }
                curr = head;
                while (list1 != null && list2 != null) {
                    if (list1.val > list2.val) {
                        curr.next = list2;
                        list2 = list2.next;
                    } else {
                        curr.next = list1;
                        list1 = list1.next;
                    }
                    curr = curr.next;
                }
                ListNode reside = (list1 == null? list2: list1);
                while (reside != null) {
                    curr.next = reside;
                    reside = reside.next;
                    curr = curr.next;
                }
                return head;
            }
        }

        public ListNode reverseBetween(ListNode head, int left, int right) {
            if (head.next == null || left == right) {
                return head;
            }
            ArrayList<ListNode> trans = new ArrayList<>();
            ListNode curr = head;
            int st = 1;
            while (st < left-1) {
                curr = curr.next;
                st++;
            }
            ListNode lp = curr;
            ListNode rp = curr;
            while (st < right+1) {
                rp = rp.next;
                st++;
            }
            if (left == 1) {
                trans.add(curr);
            }
            while (trans.size() < right-left+1) {
                curr = curr.next;
                trans.add(curr);
            }
            for (int i = trans.size()-1; i >= 0; i--) {
                ListNode c = trans.get(i);
                if (i != 0) {
                    c.next = trans.get(i-1);
                }
                else {
                    c.next = rp;
                }
            }
            if (left != 1) {
                lp.next = trans.getLast();
            } else {
                head = trans.getLast();
            }
            return head;
        }

        public ListNode reverseKGroup(ListNode head, int k) {
            if (head.next == null || k == 1) {
                return head;
            }
            ListNode curr = head;
            ArrayList<ListNode> nodes = new ArrayList<>();
            while (curr.next != null) {
                nodes.add(curr);
                curr = curr.next;
            }
            nodes.add(curr);    // "curr" now points to the last node
            int rounds = nodes.size()/k;
            int cr = 1;
            head = nodes.get(k-1);
            ListNode ed = nodes.getFirst();
            while (cr <= rounds) {
                int si = cr*k-1;
                int ei = (cr-1)*k;
                for (int i = si; i >= ei; i--) {
                    ListNode c = nodes.get(i);
                    if (i != ei) {
                        c.next = nodes.get(i-1);
                    } else {
                        c.next = null;
                    }
                }
                if (cr != 1) {
                    ed.next = nodes.get(si);
                    ed = nodes.get(ei);
                }
                cr++;
            }
            // "ed" now points to the last node in the reversed set
            if (nodes.size()%k != 0) {
                ed.next = nodes.get((cr-1)*k);
            }
            return head;
        }

        public ListNode removeNthFromEnd(ListNode head, int n) {
            if (head.next == null) {
                return null;
            }
            ArrayList<ListNode> listNodes = new ArrayList<>();
            ListNode curr = head;
            while (curr != null) {
                listNodes.add(curr);
                curr = curr.next;
            }
            int di = listNodes.size()-n;
            if (di == 0) {
                head = head.next;
            } else if (di == listNodes.size()-1) {
                listNodes.get(listNodes.size()-2).next = null;
            } else {
                listNodes.get(di-1).next = listNodes.get(di+1);
            }
            return head;
        }

        public ListNode deleteDuplicates(ListNode head) {
            if (head == null || head.next == null) {
                return head;
            }
            ListNode curr = head;
            ArrayList<ListNode> nodes = new ArrayList<>();
            while (curr != null) {
                if (!nodes.isEmpty()) {
                    if (curr.val == nodes.getLast().val) {
                        while (curr != null) {
                            if (curr.val != nodes.getLast().val) {
                                break;
                            } else {
                                curr = curr.next;
                            }
                        }
                        nodes.removeLast();
                        if (!nodes.isEmpty()) {
                            nodes.getLast().next = curr;
                            if (curr == null) {
                                break;
                            } else {
                                nodes.add(curr);
                            }
                        } else {
                            if (curr == null) {
                                return null;
                            } else {
                                nodes.add(curr);
                            }
                        }
                    } else {
                        nodes.add(curr);
                    }
                } else {
                    nodes.add(curr);
                }
                curr = curr.next;
            }
            return nodes.getFirst();
        }

        public ListNode rotateRight(ListNode head, int k) {
            if (head == null || head.next == null || k == 0) {
                return head;
            }
            ArrayList<ListNode> nodes = new ArrayList<>();
            ListNode curr = head;
            while (curr != null) {
                nodes.add(curr);
                curr = curr.next;
            }
            if (k%nodes.size() == 0) {
                return head;
            }
            int rk = k%nodes.size();
            int hi = nodes.size()-rk;
            nodes.getLast().next = head;
            nodes.get(hi-1).next = null;
            return nodes.get(hi);
        }

        public ListNode partition(ListNode head, int x) {
            if (head == null || head.next == null) {
                return head;
            }
            ListNode curr = head;
            ArrayList<ListNode> nodes = new ArrayList<>();
            while (curr != null) {
                nodes.add(curr);
                curr = curr.next;
            }
            ListNode s1 = null;
            for (ListNode listNode: nodes) {
                if (listNode.val < x) {
                    if (s1 == null) {
                        s1 = listNode;
                        head = s1;
                    } else {
                        s1.next = listNode;
                        s1 = listNode;
                    }
                }
            }
            if (s1 == null) {
                return head;
            }
            for (ListNode listNode: nodes) {
                if (listNode.val >= x) {
                    s1.next = listNode;
                    s1 = listNode;
                }
            }
            s1.next = null;
            return head;
        }

        public ListNode sortList(ListNode head) {
            if (head == null || head.next == null) return head;
            ListNode f = head;
            ListNode s = head;
            while (f.next != null && f.next.next != null) {
                f = f.next.next;
                s = s.next;
            }
            ListNode mid = s.next;
            s.next = null;
            ListNode l = sortList(head);
            ListNode r = sortList(mid);
            ListNode eh = new ListNode();
            ListNode th = eh;
            while (l != null && r != null) {
                if (l.val < r.val) {
                    th.next = l;
                    th = th.next;
                    l = l.next;
                } else {
                    th.next = r;
                    th = th.next;
                    r = r.next;
                }
            }
            th.next = (l == null) ? r : l;
            return eh.next;
        }

        public ListNode mergeKLists(ListNode[] lists) {
            // Time complexity: O(N*log(k))  N: total number of nodes  k: number of linked lists
            if (lists.length == 0) return null;
            if (lists.length == 1) return lists[0];
            int ll = lists.length/2;
            ListNode[] lp = new ListNode[ll];
            ListNode[] rp = new ListNode[lists.length-ll];
            System.arraycopy(lists, 0, lp, 0, ll);
            System.arraycopy(lists, ll, rp, 0, lists.length-ll);
            ListNode ln = mergeKLists(lp);
            ListNode rn = mergeKLists(rp);
            if (ln == null || rn == null) return (ln == null) ? rn : ln;
            ListNode head = new ListNode();
            ListNode curr = head;
            while (ln != null && rn != null) {
                if (ln.val <= rn.val) {
                    curr.next = ln;
                    ln = ln.next;
                } else {
                    curr.next = rn;
                    rn = rn.next;
                }
                curr = curr.next;
            }
            curr.next = (ln == null) ? rn : ln;
            return head.next;
        }
    }

    static class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }

        public Node copyRandomList(Node head) {
            if (head == null) {
                return null;
            }
            Node hc = new Node(head.val);
            Node hcc = hc;
            Node hc2 = hc;
            ArrayList<Node> arrayList = new ArrayList<>();   // note: there's another solution that achieves O(1) in space
            arrayList.add(head);
            ArrayList<Node> arrayList2 = new ArrayList<>();
            arrayList2.add(hc);
            Node h2 = head;
            head = head.next;
            while (head != null) {
                arrayList.add(head);
                hc.next = new Node(head.val);
                hc = hc.next;
                arrayList2.add(hc);
                head = head.next;
            }
            while (h2 != null) {
                if (h2.random != null) {
                    hcc.random = arrayList2.get(arrayList.indexOf(h2.random));
                }
                h2 = h2.next;
                hcc = hcc.next;
            }
            return hc2;
        }
    }

    static class LRUCache {
        // O(1) time requirements
        // Doubly linked list approach
        private static class Node {
            int key;
            int value;
            Node prev;
            Node next;

            Node(int key, int value) {
                this.key = key;
                this.value = value;
            }
        }

        private final int capacity;
        private final HashMap<Integer, Node> map;
        private final Node head;
        private final Node tail;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            this.map = new HashMap<>();
            this.head = new Node(-1, -1);
            this.tail = new Node(-1, -1);
            head.next = tail;
            tail.prev = head;
        }

        public int get(int key) {
            if (!map.containsKey(key)) {
                return -1;
            }
            Node node = map.get(key);
            remove(node);
            insertToHead(node);
            return node.value;
        }

        public void put(int key, int value) {
            if (map.containsKey(key)) {
                remove(map.get(key));
            }
            if (map.size() == capacity) {
                remove(tail.prev);
            }
            Node newNode = new Node(key, value);
            map.put(key, newNode);
            insertToHead(newNode);
        }

        private void remove(Node node) {
            map.remove(node.key);
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }

        private void insertToHead(Node node) {
            map.put(node.key, node);
            node.next = head.next;
            node.prev = head;
            head.next.prev = node;
            head.next = node;
        }
    }

    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int val) {
            this.val = val;
        }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }

        public int maxDepth(TreeNode root) {
            if (root == null) {
                return 0;
            }
            return Math.max(maxDepth(root.left), maxDepth(root.right))+1;
        }

        public boolean isSameTree(TreeNode p, TreeNode q) {
            if (p == null && q == null) {
                return true;
            } else if (p == null || q == null) {
                return false;
            }
            if (p.val != q.val) {
                return false;
            }
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }

        public TreeNode invertTree(TreeNode root) {
            if (root == null) {
                return null;
            }
            TreeNode temp = root.left;
            root.left = invertTree(root.right);
            root.right = invertTree(temp);
            return root;
        }

        private boolean mirror(TreeNode p, TreeNode q) {
            if (p == null && q == null) {
                return true;
            } else if (p == null || q == null) {
                return false;
            }
            if (p.val != q.val) {
                return false;
            }
            return mirror(p.left, q.right) && mirror(p.right, q.left);
        }

        public boolean isSymmetric(TreeNode root) {
            return mirror(root.left, root.right);
        }

        private TreeNode buildTreeHelper(int[] preorder, int preStart, int preEnd, int[] inorder, int inStart, int inEnd, Map<Integer, Integer> inorderIndexMap) {
            if (preStart > preEnd || inStart > inEnd) {
                return null;
            }
            // The first element in the preorder range is the root
            int rootVal = preorder[preStart];
            TreeNode root = new TreeNode(rootVal);
            // Find the index of the root in the inorder array
            int inorderRootIndex = inorderIndexMap.get(rootVal);
            // Number of elements in the left subtree
            int leftSubtreeSize = inorderRootIndex - inStart;
            // Recursively build the left and right subtrees
            root.left = buildTreeHelper(preorder, preStart+1, preStart+leftSubtreeSize, inorder, inStart, inorderRootIndex-1, inorderIndexMap);
            root.right = buildTreeHelper(preorder, preStart+leftSubtreeSize+1, preEnd, inorder, inorderRootIndex+1, inEnd, inorderIndexMap);
            return root;
        }

        public TreeNode buildTree(int[] preorder, int[] inorder) {
            // Preorder: root->left->right  Inorder: left->root->right
            // Create a map to store the index of each value in the inorder array
            Map<Integer, Integer> inorderIndexMap = new HashMap<>();
            for (int i = 0; i < inorder.length; i++) {
                inorderIndexMap.put(inorder[i], i);
            }
            // Helper function to build the tree recursively
            return buildTreeHelper(preorder, 0, preorder.length-1, inorder, 0, inorder.length-1, inorderIndexMap);
        }

        private TreeNode buildTreeHelper_II(int[] postorder, int postStart, int postEnd, int[] inorder, int inStart, int inEnd, Map<Integer, Integer> inorderIndexMap) {
            if (postStart > postEnd || inStart > inEnd) {
                return null;
            }
            // The last element in the postorder range is the root
            int rootVal = postorder[postEnd];
            TreeNode root = new TreeNode(rootVal);
            int inorderRootIndex = inorderIndexMap.get(rootVal);
            int leftSubtreeSize = inorderRootIndex - inStart;
            root.left = buildTreeHelper_II(postorder, postStart, postStart+leftSubtreeSize-1, inorder, inStart, inorderRootIndex-1, inorderIndexMap);
            root.right = buildTreeHelper_II(postorder, postStart+leftSubtreeSize, postEnd-1, inorder, inorderRootIndex+1, inEnd, inorderIndexMap);
            return root;
        }

        public TreeNode buildTree_II(int[] inorder, int[] postorder) {
            // Postorder: left->right->root
            Map<Integer, Integer> inorderIndexMap = new HashMap<>();
            for (int i = 0; i < inorder.length; i++) {
                inorderIndexMap.put(inorder[i], i);
            }
            return buildTreeHelper_II(postorder, 0, postorder.length-1, inorder, 0, inorder.length-1, inorderIndexMap);
        }

        public void flatten(TreeNode root) {
            if (root == null) {
                return;
            }
            LinkedList<TreeNode> queue = new LinkedList<>();
            queue.push(root);
            TreeNode prev = root;
            while (!queue.isEmpty()) {
                TreeNode curr = queue.poll();
                int lf = 0;
                if (curr.left != null) {
                    queue.push(curr.left);
                    lf = 1;
                }
                if (curr.right != null) {
                    if (lf == 1) {
                        queue.add(1, curr.right);
                    } else {
                        queue.push(curr.right);
                    }
                }
                if (curr != prev) {
                    prev.left = null;
                    prev.right = curr;
                    prev = curr;
                }
            }
        }

        public boolean hasPathSum(TreeNode root, int targetSum) {
            if (root == null) {
                return false;
            }
            if (root.val == targetSum && root.left == null && root.right == null) {
                return true;
            }
            return hasPathSum(root.left, targetSum - root.val)
                    || hasPathSum(root.right, targetSum - root.val);
        }

        private int dfs(TreeNode node, int currentNumber) {
            if (node == null) {
                return 0;
            }
            // Update the current number by shifting the previous number to the left and adding the current node's value
            currentNumber = currentNumber * 10 + node.val;
            // If it is a leaf node, return the current number
            if (node.left == null && node.right == null) {
                return currentNumber;
            }
            // Recursively calculate the sum for left and right subtrees
            int leftSum = dfs(node.left, currentNumber);
            int rightSum = dfs(node.right, currentNumber);
            // Return the sum of the left and right subtrees
            return leftSum + rightSum;
        }
        public int sumNumbers(TreeNode root) {
            return dfs(root, 0);
        }

        private int maxSum = Integer.MIN_VALUE;
        private int maxPathSumHelper(TreeNode node) {
            if (node == null) {
                return 0;
            }
            // Calculate the maximum path sum of the left and right subtrees
            int leftMax = Math.max(0, maxPathSumHelper(node.left));
            int rightMax = Math.max(0, maxPathSumHelper(node.right));
            // Calculate the maximum path sum through the current node
            int currentMax = node.val+leftMax+rightMax;
            // Update the global maximum path sum
            maxSum = Math.max(maxSum, currentMax);
            // Return the maximum path sum that can be extended to the parent node
            return node.val+Math.max(leftMax, rightMax);
        }
        public int maxPathSum(TreeNode root) {
            maxPathSumHelper(root);
            return maxSum;
        }

        static class BSTIterator {
            ArrayList<TreeNode> iter;
            int pt = -1;

            private void traverse(TreeNode node) {
                if (node != null) {
                    traverse(node.left);
                    iter.add(node);
                    traverse(node.right);
                }
            }
            public BSTIterator(TreeNode root) {
                iter = new ArrayList<>();
                traverse(root);
            }

            public int next() {
                pt += 1;
                return iter.get(pt).val;
            }

            public boolean hasNext() {
                return pt < iter.size() - 1;
            }
        }

        private int getHeight(TreeNode curr, int flag) {
            int h = 0;
            while (curr != null) {
                h += 1;
                if (flag == 1) {
                    curr = curr.left;
                } else {
                    curr = curr.right;
                }
            }
            return h;
        }
        public int countNodes(TreeNode root) {
            if (root == null) {
                return 0;
            }
            int lh = getHeight(root, 1);          // Time complexity: O(log(n)*log(n))
            int rh = getHeight(root, 0);
            if (lh == rh) {
                return (1<<lh)-1;
            }
            return 1+countNodes(root.left)+countNodes(root.right);
        }

        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            if (root == null || root == p || root == q) {
                return root;
            }
            TreeNode lr = lowestCommonAncestor(root.left, p, q);
            TreeNode rr = lowestCommonAncestor(root.right, p, q);
            if (lr != null && rr != null) {
                return root;
            }
            return (lr == null)?rr:lr;
        }

        public List<Integer> rightSideView(TreeNode root) {
            if (root == null) {
                return new ArrayList<>(0);
            }
            List<Integer> results = new ArrayList<>();
            results.add(root.val);
            List<TreeNode> bst = new ArrayList<>();
            bst.add(root);
            while (!bst.isEmpty()) {
                List<TreeNode> lvl = new ArrayList<>(0);
                for (TreeNode tn: bst) {
                    if (tn.left != null) {
                        lvl.add(tn.left);
                    }
                    if (tn.right != null) {
                        lvl.add(tn.right);
                    }
                }
                bst.clear();
                if (!lvl.isEmpty()) {
                    results.add(lvl.getLast().val);
                    bst.addAll(lvl);
                }
            }
            return results;
        }

        public List<Double> averageOfLevels(TreeNode root) {
            List<Double> results = new ArrayList<>();
            results.add((double) root.val);
            List<TreeNode> bst = new ArrayList<>();
            bst.add(root);
            while (!bst.isEmpty()) {
                List<TreeNode> lvl = new ArrayList<>(0);
                for (TreeNode tn: bst) {
                    if (tn.left != null) {
                        lvl.add(tn.left);
                    }
                    if (tn.right != null) {
                        lvl.add(tn.right);
                    }
                }
                bst.clear();
                if (!lvl.isEmpty()) {
                    double sum = 0;
                    for (TreeNode node: lvl) {
                        sum += node.val;
                    }
                    sum /= lvl.size();
                    results.add(sum);
                    bst.addAll(lvl);
                }
            }
            return results;
        }

        public List<List<Integer>> levelOrder(TreeNode root) {
            // More clarity
            List<List<Integer>> result = new ArrayList<>();
            if (root == null) {
                return result;
            }
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            while (!queue.isEmpty()) {
                int levelSize = queue.size();
                List<Integer> currentLevel = new ArrayList<>();
                for (int i = 0; i < levelSize; i++) {
                    TreeNode currentNode = queue.poll();
                    assert currentNode != null;
                    currentLevel.add(currentNode.val);
                    if (currentNode.left != null) {
                        queue.offer(currentNode.left);
                    }
                    if (currentNode.right != null) {
                        queue.offer(currentNode.right);
                    }
                }
                result.add(currentLevel);
            }
            return result;
        }

        public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
            List<List<Integer>> result = new ArrayList<>();
            if (root == null) {
                return result;
            }
            LinkedList<TreeNode> lvl = new LinkedList<>();
            lvl.offer(root);
            int dir = 1;
            while (!lvl.isEmpty()) {
                int level_size = lvl.size();
                List<Integer> lv = new ArrayList<>();
                if (dir == 1) {
                    for (int i = 0; i <= level_size-1; i++) {
                        TreeNode curr = lvl.poll();
                        assert curr != null;
                        lv.add(curr.val);
                        if (curr.left != null) {
                            lvl.offer(curr.left);
                        }
                        if (curr.right != null) {
                            lvl.offer(curr.right);
                        }
                    }
                } else {
                    for (int i = level_size-1; i >= 0; i--) {
                        TreeNode curr = lvl.removeLast();
                        lv.add(curr.val);
                        if (curr.right != null) {
                            lvl.push(curr.right);
                        }
                        if (curr.left != null) {
                            lvl.push(curr.left);
                        }
                    }
                }
                result.add(lv);
                dir *= -1;
            }
            return result;
        }

        private int minDifference = Integer.MAX_VALUE;
        private TreeNode prev = null;
        public int getMinimumDifference(TreeNode root) {
            inOrderTraversal(root);
            return minDifference;
        }
        private void inOrderTraversal(TreeNode node) {
            if (node == null) {
                return;
            }
            inOrderTraversal(node.left);
            if (prev != null) {
                minDifference = Math.min(minDifference, node.val-prev.val);
            }
            prev = node;
            inOrderTraversal(node.right);
        }

        private int count = 0;
        private int result = 0;
        public int kthSmallest(TreeNode root, int k) {
            inOrderTraversal(root, k);
            return result;
        }
        private void inOrderTraversal(TreeNode node, int k) {
            if (node == null) {
                return;
            }
            // Traverse the left subtree
            inOrderTraversal(node.left, k);
            // Process the current node
            count++;
            if (count == k) {
                result = node.val;
                return; // Early termination once the kth smallest is found
            }
            // Traverse the right subtree
            inOrderTraversal(node.right, k);
        }

        private boolean isValidBSTHelper(TreeNode node, TreeNode[] prev) {
            if (node == null) {
                return true;
            }
            // Check the left subtree
            if (!isValidBSTHelper(node.left, prev)) {
                return false;
            }
            // Check the current node
            if (prev[0] != null && node.val <= prev[0].val) {
                return false;
            }
            // Update the previous node
            prev[0] = node;
            // Check the right subtree
            return isValidBSTHelper(node.right, prev);
        }
        public boolean isValidBST(TreeNode root) {
            // Use an array to hold the previous node during in-order traversal
            TreeNode[] prev = new TreeNode[1];
            return isValidBSTHelper(root, prev);
        }

        public TreeNode sortedArrayToBST(int[] nums) {
            if (nums.length == 1) return new TreeNode(nums[0]);
            int top = nums.length/2;
            int[] left = new int[top];
            System.arraycopy(nums, 0, left, 0, top);  // System.arraycopy(): built-in method, generally lower overhead than using a 'for' loop
            if (nums.length == 2) return new TreeNode(nums[top], sortedArrayToBST(left), null);
            int[] right = new int[nums.length-top-1];
            System.arraycopy(nums, top+1, right, 0, nums.length-top-1);
            return new TreeNode(nums[top], sortedArrayToBST(left), sortedArrayToBST(right));
        }
    }

    static class TNode {
        public int val;
        public TNode left;
        public TNode right;
        public TNode next;

        public TNode(int _val) {
            val = _val;
        }
        public TNode(int _val, TNode _left, TNode _right, TNode _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }

        public TNode connect(TNode root) {
            if (root == null) {
                return null;
            }
            Queue<TNode> queue = new LinkedList<>();
            queue.add(root);
            while (!queue.isEmpty()) {
                int size = queue.size();
                TNode prev = null;
                for (int i = 0; i < size; i++) {
                    TNode current = queue.poll();
                    if (prev != null) {
                        prev.next = current;
                    }
                    prev = current;
                    assert current != null;
                    if (current.left != null) {
                        queue.add(current.left);
                    }
                    if (current.right != null) {
                        queue.add(current.right);
                    }
                }
                prev.next = null;
            }
            return root;
        }
    }

    public void dfs(char[][] grid, int cr, int cc) {
        if (cr < 0 || cc < 0 || cr >= grid.length || cc >= grid[0].length || grid[cr][cc] == '0') {
            return;
        }
        grid[cr][cc] = '0';
        dfs(grid, cr, cc+1);
        dfs(grid, cr+1, cc);
        dfs(grid, cr, cc-1);
        dfs(grid, cr-1, cc);
    }
    public int numIslands(char[][] grid) {
        int islands = 0;
        for (int r = 0; r < grid.length; r++) {
            for (int c = 0; c < grid[0].length; c++) {
                if (grid[r][c] == '1') {
                    islands += 1;
                    dfs(grid, r, c);
                }
            }
        }
        return islands;
    }

    private void dfs2(char[][] board, int row, int col) {
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length || board[row][col] != 'O') {
            return;
        }
        board[row][col] = '#'; // Mark this cell as visited
        dfs2(board, row + 1, col);
        dfs2(board, row - 1, col);
        dfs2(board, row, col + 1);
        dfs2(board, row, col - 1);
    }
    public void solve(char[][] board) {
        int rows = board.length;
        int cols = board[0].length;

        // Step 1: Mark all 'O's connected to the border
        for (int i = 0; i < rows; i++) {
            if (board[i][0] == 'O') {
                dfs2(board, i, 0);
            }
            if (board[i][cols - 1] == 'O') {
                dfs2(board, i, cols - 1);
            }
        }
        for (int j = 0; j < cols; j++) {
            if (board[0][j] == 'O') {
                dfs2(board, 0, j);
            }
            if (board[rows - 1][j] == 'O') {
                dfs2(board, rows - 1, j);
            }
        }

        // Step 2: Capture surrounded regions
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                } else if (board[i][j] == '#') {
                    board[i][j] = 'O';
                }
            }
        }
    }

    static class GNode {
        public int val;
        public List<GNode> neighbors;

        public GNode(int val) {
            this.val = val;
            this.neighbors = new ArrayList<>();
        }

        private final HashMap<GNode, GNode> map = new HashMap<>();
        public GNode cloneGraph(GNode node) {
            if (node == null) {
                return null;
            }
            if (map.containsKey(node)) {
                return map.get(node);
            }
            GNode copy = new GNode(node.val);
            map.put(node, copy);
            for (GNode n: node.neighbors) {
                copy.neighbors.add(cloneGraph(n));
            }
            return copy;
        }
    }

    private double dfs(String start, String end, Map<String, Map<String, Double>> graph, Set<String> visited, double product) {
        if (start.equals(end)) {
            return product;
        }
        visited.add(start);
        for (Map.Entry<String, Double> entry : graph.get(start).entrySet()) {
            String next = entry.getKey();
            double value = entry.getValue();
            if (!visited.contains(next)) {
                double result = dfs(next, end, graph, visited, product * value);
                if (result != -1.0) {
                    return result;
                }
            }
        }
        return -1.0;
    }
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        // Create a graph to store the equations
        Map<String, Map<String, Double>> graph = new HashMap<>();
        // Build the graph
        for (int i = 0; i < equations.size(); i++) {
            String A = equations.get(i).get(0);
            String B = equations.get(i).get(1);
            double value = values[i];
            graph.putIfAbsent(A, new HashMap<>());
            graph.putIfAbsent(B, new HashMap<>());
            graph.get(A).put(B, value);
            graph.get(B).put(A, 1.0 / value);
        }
        // Process each query
        double[] results = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            String C = queries.get(i).get(0);
            String D = queries.get(i).get(1);
            if (!graph.containsKey(C) || !graph.containsKey(D)) {
                results[i] = -1.0;
            } else {
                results[i] = dfs(C, D, graph, new HashSet<>(), 1.0);
            }
        }
        return results;
    }

    private boolean isCyclic(int node, List<List<Integer>> adjList, int[] visited) {
        if (visited[node] == 1) {   // If the node is being visited (part of the recursion stack)
            return true;
        }
        if (visited[node] == 2) {   // If the node has been fully visited
            return false;
        }
        visited[node] = 1;   // Mark the node as being visited
        for (int neighbor : adjList.get(node)) {
            if (isCyclic(neighbor, adjList, visited)) {
                return true;
            }
        }
        visited[node] = 2;   // Mark the node as fully visited
        return false;
    }
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // Create an adjacency list to represent the graph
        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            adjList.add(new ArrayList<>());
        }
        // Build the graph
        for (int[] prerequisite : prerequisites) {
            adjList.get(prerequisite[1]).add(prerequisite[0]);
        }
        // Visited array to keep track of visited nodes
        int[] visited = new int[numCourses];
        // DFS to detect cycle
        for (int i = 0; i < numCourses; i++) {
            if (visited[i] == 0) { // If the node has not been visited
                if (isCyclic(i, adjList, visited)) {
                    return false;
                }
            }
        }
        return true;
    }

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        // Kahn's algorithm
        // Useful for scheduling tasks where certain tasks must be completed before others can begin
        List<List<Integer>> adjList = new ArrayList<>();      // construct an adjacency list to represent the graph
        for (int i = 0; i <= numCourses-1; i++) {
            adjList.add(new ArrayList<>());
        }
        int[] inDegree = new int[numCourses];
        for (int[] preq: prerequisites) {
            adjList.get(preq[1]).add(preq[0]);          // get the 'higher' topological neighbor
            inDegree[preq[0]] += 1;                     // get the in-degree of each vertex
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i <= numCourses-1; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        List<Integer> topo = new ArrayList<>();
        while (!queue.isEmpty()) {
            int course = queue.poll();
            topo.add(course);
            for (int neighbor: adjList.get(course)) {   // no error even if the returned ArrayList is empty
                inDegree[neighbor] -= 1;
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        if (topo.size() == numCourses) {
            int[] order = new int[numCourses];
            for (int i = 0; i <= numCourses-1; i++) {
                order[i] = topo.get(i);
            }
            return order;
        } else {                        // there exists a cycle
            return new int[0];
        }
    }

    private int[] getCoordinate(int square, int size) {
        // Helper method that converts the number of a square to its coordinate.
        int[] coordinate = new int[2];
        int r = size-1-(square-1)/size;
        int c;
        if (size%2 != r%2) {           // from left to right
            c = (square-1)%size;
        } else {                       // from right to left
            c = size-1-(square-1)%size;
        }
        coordinate[0] = r;
        coordinate[1] = c;
        return coordinate;
    }
    public int snakesAndLadders(int[][] board) {
        int n = board.length;
        Queue<int[]> queue = new LinkedList<>();
        Set<Integer> visit = new HashSet<>();
        queue.offer(new int[]{1, 0});
        visit.add(1);
        while (!queue.isEmpty()) {
            int[] curr = queue.poll();
            int sq = curr[0];
            int move = curr[1];
            for (int i = 1; i <= 6; i++) {
                int next = sq+i;
                if (next > n*n) break;
                int[] nc = getCoordinate(next, n);
                int r = nc[0];
                int c = nc[1];
                if (board[r][c] != -1) next = board[r][c];
                if (next == n*n) return move+1;
                if (!visit.contains(next)) {
                    queue.offer(new int[]{next, move+1});
                    visit.add(next);
                }
            }
        }
        return -1;
    }

    public int minMutation(String startGene, String endGene, String[] bank) {
        if (bank.length == 0) return -1;
        Set<String> bankset = new HashSet<>(Arrays.asList(bank));
        if (!bankset.contains(endGene)) return -1;
        if (startGene.equals(endGene)) return 0;
        Queue<String> queue = new LinkedList<>();
        Map<String, Integer> map = new HashMap<>();
        Set<String> set = new HashSet<>();
        queue.offer(startGene);
        map.put(startGene, 0);
        set.add(startGene);
        char[] genes = new char[] {'A', 'C', 'G', 'T'};
        while (!queue.isEmpty()) {
            String curr = queue.poll();
            for (int i = 0; i <= startGene.length()-1; i++) {
                char[] chars = curr.toCharArray();
                for (char g: genes) {
                    chars[i] = g;
                    String newString = new String(chars);
                    if (newString.equals(endGene)) return map.get(curr)+1;
                    if (!bankset.contains(newString)) continue;
                    if (!set.contains(newString)) {
                        queue.offer(newString);
                        map.put(newString, map.get(curr)+1);
                        set.add(newString);
                    }
                }
            }
        }
        return -1;
    }

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        // Note: convert the List<> to the HashSet<> first in case of large input.
        // Difference in time complexity of the .contains() method: List<>: O(n), HashSet<>: O(1)
        Set<String> wordSet = new HashSet<>(wordList);
        if (!wordSet.contains(endWord)) return 0;
        Queue<String> queue = new LinkedList<>();
        Map<String, Integer> map = new HashMap<>();
        Set<String> visit = new HashSet<>();
        queue.offer(beginWord);
        map.put(beginWord, 1);
        visit.add(beginWord);
        while (!queue.isEmpty()) {
            String curr = queue.poll();
            for (int i = 0; i <= curr.length()-1; i++) {
                char[] chars = curr.toCharArray();
                for (int j = 1; j <= 25; j++) {
                    chars[i] = (char) ((chars[i]>121) ? chars[i]-25 : chars[i]+1);
                    String newWord = new String(chars);
                    if (!wordSet.contains(newWord)) continue;
                    if (newWord.equals(endWord)) return map.get(curr)+1;
                    if (!visit.contains(newWord)) {
                        queue.offer(newWord);
                        map.put(newWord, map.get(curr)+1);
                        visit.add(newWord);
                    }
                }
            }
        }
        return 0;
    }

    static class TrieNode {
        TrieNode[] children;
        boolean EOW;

        public TrieNode() {
            children = new TrieNode[26];
            EOW = false;
        }
    }
    static class Trie {
        // A more time-efficient approach: O(n) for 'insert()', 'search()', and 'startsWith()'.
        TrieNode root;

        public Trie() {
            root = new TrieNode();
        }
        public void insert(String word) {
            TrieNode curr = root;
            for (char c: word.toCharArray()) {
                int index = c-'a';
                if (curr.children[index] == null) curr.children[index] = new TrieNode();
                curr = curr.children[index];
            }
            curr.EOW = true;
        }
        public boolean search(String word) {
            TrieNode curr = root;
            for (char c: word.toCharArray()) {
                int index = c-'a';
                if (curr.children[index] == null) return false;
                curr = curr.children[index];
            }
            return curr.EOW;
        }
        public boolean startsWith(String prefix) {
            TrieNode curr = root;
            for (char c: prefix.toCharArray()) {
                int index = c-'a';
                if (curr.children[index] == null) return false;
                curr = curr.children[index];
            }
            return true;
        }
    }
    static class WordDictionary {
        TrieNode root;

        public WordDictionary() {
            root = new TrieNode();
        }

        public void addWord(String word) {
            TrieNode curr = root;
            for (char c: word.toCharArray()) {
                int index = c-'a';
                if (curr.children[index] == null) curr.children[index] = new TrieNode();
                curr = curr.children[index];
            }
            curr.EOW = true;
        }

        private boolean wildcard(String word, TrieNode node, int index) {
            if (index == word.length()) return node.EOW;
            char c = word.charAt(index);
            if (c == '.') {
                for (TrieNode trieNode: node.children) {
                    if (trieNode != null && wildcard(word, trieNode, index+1)) return true;
                }
                return false;
            } else {
                int pos = c-'a';
                if (node.children[pos] == null) return false;
                return wildcard(word, node.children[pos], index+1);
            }
        }
        public boolean search(String word) {
            return wildcard(word, root, 0);
        }
    }

    static class WordTrie {
        String word;
        WordTrie[] wordTries;

        WordTrie() {
            wordTries = new WordTrie[26];
        }
    }
    private WordTrie buildWT(String[] words) {
        WordTrie wordTrie = new WordTrie();
        for (String word: words) {
            WordTrie node = wordTrie;
            for (char c: word.toCharArray()) {
                int index = c-'a';
                if (node.wordTries[index] == null) node.wordTries[index] = new WordTrie();
                node = node.wordTries[index];
            }
            node.word = word;
        }
        return wordTrie;
    }
    private void DFS(char[][] board, int row, int col, WordTrie trie, List<String> result) {
        char c = board[row][col];
        if (c == '#' || trie.wordTries[c-'a'] == null) return;
        trie = trie.wordTries[c-'a'];
        if (trie.word != null) {
            result.add(trie.word);
            trie.word = null;
        }
        board[row][col] = '#';
        if (row > 0) DFS(board, row-1, col, trie, result);
        if (col > 0) DFS(board, row, col-1, trie, result);
        if (row < board.length-1) DFS(board, row+1, col, trie, result);
        if (col < board[0].length-1) DFS(board, row, col+1, trie, result);
        board[row][col] = c;
    }
    public List<String> findWords(char[][] board, String[] words) {
        WordTrie root = buildWT(words);
        List<String> list = new ArrayList<>();
        for (int i = 0; i <= board.length-1; i++) {
            for (int j = 0; j <= board[0].length-1; j++) {
                DFS(board, i, j, root, list);
            }
        }
        return list;
    }

    private char[] getLetters(char d) {
        return switch (d) {
            case '2' -> new char[]{'a', 'b', 'c'};
            case '3' -> new char[]{'d', 'e', 'f'};
            case '4' -> new char[]{'g', 'h', 'i'};
            case '5' -> new char[]{'j', 'k', 'l'};
            case '6' -> new char[]{'m', 'n', 'o'};
            case '7' -> new char[]{'p', 'q', 'r', 's'};
            case '8' -> new char[]{'t', 'u', 'v'};
            default -> new char[]{'w', 'x', 'y', 'z'};
        };
    }
    public List<String> letterCombinations(String digits) {
        List<String> list = new ArrayList<>(0);
        if (digits.isEmpty()) return list;
        List<char[]> letters = new ArrayList<>();
        int n = 1;
        for (char d: digits.toCharArray()) {
            char[] ds = getLetters(d);
            letters.add(ds);
            n *= ds.length;
        }
        int[] pos = new int[letters.size()];
        int i = 1;
        while (i <= n) {
            StringBuilder stringBuilder = new StringBuilder();
            for (int k = 0; k <= letters.size()-1; k++) stringBuilder.append(letters.get(k)[pos[k]]);
            list.add(stringBuilder.toString());
            int addon = 0;
            for (int p = letters.size()-1; p >= 0; p--) {
                if (p == letters.size()-1) {
                    pos[p] += 1;
                } else {
                    if (addon == 0) break;
                    pos[p] += 1;
                    addon = 0;
                }
                if (pos[p] == letters.get(p).length) {
                    pos[p] = 0;
                    addon = 1;
                }
            }
            i++;
        }
        return list;
    }

    private void backtrack(List<List<Integer>> res, List<Integer> curr, int s, int e, int u) {
        if (curr.size() == u) {
            res.add(new ArrayList<>(curr));        // create a deep copy
        } else {
            for (int i = s; i <= e; i++) {
                curr.add(i);
                backtrack(res, curr, i+1, e, u);
                curr.removeLast();
            }
        }
    }
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> lists = new ArrayList<>();
        backtrack(lists, new ArrayList<>(), 1, n, k);
        return lists;
    }

    private void backtrack(List<List<Integer>> res, List<Integer> curr, int[] ints) {
        if (curr.size() == ints.length) {
            res.add(new ArrayList<>(curr));
        } else {
            for (int i = 0; i <= ints.length-1; i++) {
                if (!curr.contains(ints[i])) {
                    curr.add(ints[i]);
                    backtrack(res, curr, ints);
                    curr.removeLast();
                }
            }
        }
    }
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> lists = new ArrayList<>();
        backtrack(lists, new ArrayList<>(), nums);
        return lists;
    }

    private void backtrack(List<List<Integer>> res, List<Integer> curr, int[] cands, int resid, int s) {
        if (resid == 0) {
            res.add(new ArrayList<>(curr));
        } else {
            for (int i = s; i <= cands.length-1; i++) {
                int candidate = cands[i];
                if (resid >= candidate){
                    curr.add(candidate);
                    backtrack(res, curr, cands, resid-candidate, i);
                    curr.removeLast();
                }
            }
        }
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> lists = new ArrayList<>();
        backtrack(lists, new ArrayList<>(), candidates, target, 0);
        return lists;
    }

    private int countNQueens(int n, int row, int column, int diagonal, int anti_diagonal) {
        if (n == row) return 1;          // found a solution
        int avails = ((1<<n)-1)&(~(column|diagonal|anti_diagonal));  // (1<<n)-1: all positions in a row(n-bit 1),
                                                                     // column|diagonal|anti_diagonal: all positions being attacked(marked as bit 1)
        int sols = 0;
        while (avails != 0) {
            int pos = avails&(-avails);    // extracts the least significant bit -> the right most available position
            avails = avails&(avails-1);    // removes this available position
            sols += countNQueens(n,
                    row+1,
                    column|pos,
                    (diagonal|pos)>>1,
                    (anti_diagonal|pos)<<1);
        }   // move to the next row, update the columns, diagonals, and anti diagonals being attacked
        return sols;
    }
    public int totalNQueens(int n) {
        return countNQueens(n, 0, 0, 0, 0);
    }

    private void backtrack(List<String> res, StringBuilder builder, int l, int r) {
        if (l == 0 && r == 0) {
            res.add(String.valueOf(builder));
            return;
        }
        if (l == r) {
            builder.append('(');
            backtrack(res, builder, l-1, r);
        } else {
            if (l != 0) {
                StringBuilder stringBuilder = new StringBuilder(builder);
                builder.append('(');
                backtrack(res, builder, l - 1, r);
                builder = stringBuilder;
            }
            builder.append(')');
            backtrack(res, builder, l, r-1);
        }
    }
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        if (n == 1) {
            result.add("()");
        } else {
            StringBuilder stringBuilder = new StringBuilder();
            backtrack(result, stringBuilder, n, n);
        }
        return result;
    }

    private boolean backtrack(int pos, int r, int c, String w, char[][] b, int[][] m) {
        if (r >= 1 && m[r-1][c] != 1) {
            if (b[r-1][c] == w.charAt(pos)) {
                m[r-1][c] = 1;
                if (pos == w.length()-1) return true;
                if (backtrack(pos+1, r-1, c, w, b, m)) return true;
                m[r-1][c] = 0;
            }
        }
        if (c <= b[0].length-2 && m[r][c+1] != 1) {
            if (b[r][c+1] == w.charAt(pos)) {
                m[r][c+1] = 1;
                if (pos == w.length()-1) return true;
                if (backtrack(pos+1, r, c+1, w, b, m)) return true;
                m[r][c+1] = 0;
            }
        }
        if (r <= b.length-2 && m[r+1][c] != 1) {
            if (b[r+1][c] == w.charAt(pos)) {
                m[r+1][c] = 1;
                if (pos == w.length()-1) return true;
                if (backtrack(pos+1, r+1, c, w, b, m)) return true;
                m[r+1][c] = 0;
            }
        }
        if (c >= 1 && m[r][c-1] != 1) {
            if (b[r][c-1] == w.charAt(pos)) {
                m[r][c-1] = 1;
                if (pos == w.length()-1) return true;
                if (backtrack(pos+1, r, c-1, w, b, m)) return true;
                m[r][c-1] = 0;
            }
        }
        return false;
    }
    public boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        if (m*n < word.length()) return false;
        int[][] visit = new int[m][n];
        for (int i = 0; i <= m-1; i++) {
            for (int j = 0; j <= n-1; j++) {
                if (board[i][j] == word.charAt(0)) {
                    if (word.length() == 1) return true;
                    visit[i][j] = 1;
                    if (backtrack(1, i, j, word, board, visit)) return true;
                    visit[i][j] = 0;
                }
            }
        }
        return false;
    }

    static class QNode {
        public boolean val;
        public boolean isLeaf;
        public QNode topLeft;
        public QNode topRight;
        public QNode bottomLeft;
        public QNode bottomRight;

        public QNode(boolean val, boolean isLeaf) {
            this.val = val;
            this.isLeaf = isLeaf;
            this.topLeft = null;
            this.topRight = null;
            this.bottomLeft = null;
            this.bottomRight = null;
        }
        public QNode(boolean val, boolean isLeaf, QNode topLeft, QNode topRight, QNode bottomLeft, QNode bottomRight) {
            this.val = val;
            this.isLeaf = isLeaf;
            this.topLeft = topLeft;
            this.topRight = topRight;
            this.bottomLeft = bottomLeft;
            this.bottomRight = bottomRight;
        }

        public QNode construct(int[][] grid) {
            if (grid.length == 1) return new QNode(grid[0][0] == 1, true);
            int flag = grid[0][0];
            boolean ini = true;
            for (int i = 0; i <= grid.length - 1; i++) {
                for (int j = 0; j <= grid[0].length - 1; j++) {
                    if (grid[i][j] != flag) {
                        ini = false;
                        break;
                    }
                }
            }
            if (ini) return new QNode(grid[0][0] == 1, true);
            int nl = grid.length / 2;
            int[][] tl = new int[nl][nl];
            int[][] tr = new int[nl][nl];
            int[][] bl = new int[nl][nl];
            int[][] br = new int[nl][nl];
            for (int m = 0; m <= nl - 1; m++) {
                for (int n = 0; n <= nl - 1; n++) {
                    tl[m][n] = grid[m][n];
                    tr[m][n] = grid[m][n + nl];
                    bl[m][n] = grid[m + nl][n];
                    br[m][n] = grid[m + nl][n + nl];
                }
            }
            return new QNode(true, false, construct(tl), construct(tr), construct(bl), construct(br));
        }
    }

    public int maxSubArray(int[] nums) {
        // Kadane's algorithm
        // Time complexity: O(n)
        if (nums.length == 1)
            return nums[0];
        int ms = nums[0];
        int cm = ms;
        for (int i = 1; i <= nums.length - 1; i++) {
            cm = Math.max(cm + nums[i], nums[i]);
            if (cm > ms)
                ms = cm;
        }
        return ms;
    }

    private int msKadane(int[] n) {
        int ms = n[0];
        int cm = ms;
        for (int i = 1; i <= n.length-1; i++) {
            cm = Math.max(cm+n[i], n[i]);
            if (cm > ms) ms = cm;
        }
        return ms;
    }
    public int maxSubarraySumCircular(int[] nums) {
        if (nums.length == 1) return nums[0];
        // Edge case: all elements are negative
        boolean allnega = true;
        for (int n: nums) {
            if (n >= 0) {
                allnega = false;
                break;
            }
        }
        if (allnega) {
            int mn = nums[0];
            for (int n: nums) mn = Math.max(mn, n);
            return mn;
        }
        // Kadane's algorithm in a circular array
        int mcs = msKadane(nums);
        int[] inv = new int[nums.length];
        for (int i = 0; i <= nums.length-1; i++) inv[i] = -nums[i];
        int mins = -msKadane(inv);    // minimum sum of a subarray = negative of the maximum sum in the inverted array
        int total = 0;
        for (int i = 0; i <= nums.length-1; i++) total += nums[i];
        // If the maximum is obtained in a circular subarray, its sum will be (total sum of the array-minimum sum of a subarray).
        mcs = Math.max(mcs, total-mins);
        return mcs;
    }

    public int searchInsert(int[] nums, int target) {
        // Standard binary search
        int lp = 0;
        int rp = nums.length-1;
        while (lp <= rp) {
            int mid = (lp+rp)/2;
            if (nums[mid] == target) {
                return mid;
            } else if (target > nums[mid]) {
                lp = mid+1;
            } else {
                rp = mid-1;
            }
        }
        return lp;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int lp = 0;
        int rp = m*n-1;
        while (lp <= rp) {
            int mid = (lp+rp)/2;
            int r = mid/n;
            int c = mid%n;
            if (matrix[r][c] == target) {
                return true;
            } else if (target > matrix[r][c]) {
                lp = mid+1;
            } else {
                rp = mid-1;
            }
        }
        return false;
    }

    public int findPeakElement(int[] nums) {
        int lp = 0;
        int rp = nums.length-1;
        while (lp < rp) {
            int mid = (lp+rp)/2;
            if (nums[mid] > nums[mid+1]) {
                rp = mid;
            } else {
                lp = mid+1;
            }
        }
        return lp;
    }

    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            // Check if mid is the target
            if (nums[mid] == target) {
                return mid;
            }
            // Determine which part is sorted
            if (nums[left] <= nums[mid]) { // Left part is sorted
                if (nums[left] <= target && target < nums[mid]) {
                    // Target is in the left sorted part
                    right = mid - 1;
                } else {
                    // Target is in the right part
                    left = mid + 1;
                }
            } else { // Right part is sorted
                if (nums[mid] < target && target <= nums[right]) {
                    // Target is in the right sorted part
                    left = mid + 1;
                } else {
                    // Target is in the left part
                    right = mid - 1;
                }
            }
        }
        // Target is not found
        return -1;
    }

    private int findEdge(int[] nums, int target, int flag) {
        int lp = 0;
        int rp = nums.length-1;
        while (lp <= rp) {
            int mid = (lp+rp)/2;
            if (nums[mid] < target) {
                lp = mid+1;
            } else if (nums[mid] > target) {
                rp = mid-1;
            } else {
                if (flag == 0) {
                    if (mid == 0) {
                        return 0;
                    } else {
                        if (nums[mid-1] < target) {
                            return mid;
                        } else {
                            rp = mid-1;
                        }
                    }
                } else {
                    if (mid == nums.length-1) {
                        return nums.length-1;
                    } else {
                        if (nums[mid+1] > target) {
                            return mid;
                        } else {
                            lp = mid+1;
                        }
                    }
                }
            }
        }
        return -1;
    }
    public int[] searchRange(int[] nums, int target) {
        if (nums.length == 0) return new int[]{-1, -1};
        if (target > nums[nums.length-1] || target < nums[0]) return new int[]{-1, -1};
        int[] res = new int[2];
        int le = findEdge(nums, target, 0);
        if (le == -1) return new int[]{-1, -1};
        res[0] = le;
        res[1] = findEdge(nums, target, 1);
        return res;
    }

    public int findMin(int[] nums) {
        if (nums.length == 1)
            return nums[0];
        if (nums[0] < nums[nums.length - 1])
            return nums[0];
        if (nums.length == 2)
            return nums[1];
        int lp = 0;
        int rp = nums.length - 1;
        while (lp <= rp) {
            if (rp - lp >= 2) {
                int mid = (lp + rp) / 2;
                if (nums[mid - 1] < nums[mid] && nums[mid] < nums[mid + 1]) {
                    if (nums[mid] < nums[0]) {
                        rp = mid - 1;
                    } else {
                        lp = mid + 1;
                    }
                } else if (nums[mid] > nums[mid - 1] && nums[mid] > nums[mid + 1]) {
                    return nums[mid + 1];
                } else {
                    return nums[mid];
                }
            } else {
                break;
            }
        }
        return Math.min(nums[lp], nums[rp]);
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length == 0 || nums2.length == 0) {
            if (nums1.length == 0) {
                if (nums2.length % 2 == 0) {
                    return ((double) nums2[(nums2.length - 1) / 2] + nums2[(nums2.length - 1) / 2 + 1]) / 2;
                } else {
                    return nums2[(nums2.length - 1) / 2];
                }
            } else {
                if (nums1.length % 2 == 0) {
                    return ((double) nums1[(nums1.length - 1) / 2] + nums1[(nums1.length - 1) / 2 + 1]) / 2;
                } else {
                    return nums1[(nums1.length - 1) / 2];
                }
            }
        }
        if (nums1.length > nums2.length) {
            return findMedianSortedArrays(nums2, nums1);
        }
        int lp = 0;
        int rp = nums1.length;
        while (lp <= rp) {
            int p1 = (lp + rp) / 2;
            int p2 = (nums1.length + nums2.length + 1) / 2 - p1;
            int lmax1 = (p1 == 0) ? Integer.MIN_VALUE : nums1[p1 - 1];
            int rmin1 = (p1 == nums1.length) ? Integer.MAX_VALUE : nums1[p1];
            int lmax2 = (p2 == 0) ? Integer.MIN_VALUE : nums2[p2 - 1];
            int rmin2 = (p2 == nums2.length) ? Integer.MAX_VALUE : nums2[p2];
            if (lmax1 <= rmin2 && lmax2 <= rmin1) {
                if ((nums1.length + nums2.length) % 2 == 0) {
                    return ((double) Math.max(lmax1, lmax2) + Math.min(rmin1, rmin2)) / 2;
                } else {
                    return Math.max(lmax1, lmax2);
                }
            } else if (lmax1 > rmin2) {
                rp = p1 - 1;
            } else {
                lp = p1 + 1;
            }
        }
        return -1;
    }

    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int n : nums) {
            minHeap.offer(n);
            if (minHeap.size() > k) {
                minHeap.poll();
            }
        }
        return minHeap.peek();
    }

    static class Project {
        int profit;
        int capital;

        Project(int profit, int capital) {
            this.profit = profit;
            this.capital = capital;
        }
    }
    public int findMaximizedCapital(int k, int w, int[] profits, int[] capital) {
        Project[] projects = new Project[profits.length];
        for (int i = 0; i < profits.length; i++) {
            projects[i] = new Project(profits[i], capital[i]);
        }
        // Sort projects(min to max) by capital required
        Arrays.sort(projects, (a, b) -> a.capital - b.capital);
        // Max-heap to store profits of projects that can be done with current capital
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
        int currentIndex = 0;
        for (int i = 0; i < k; i++) {
            // Add all projects that can be done with current capital to the max-heap
            while (currentIndex < projects.length && projects[currentIndex].capital <= w) {
                maxHeap.offer(projects[currentIndex].profit);
                currentIndex++;
            }
            // If there are projects that can be done, pick the one with the highest profit
            if (!maxHeap.isEmpty()) {
                w += maxHeap.poll();
            } else {
                // If no projects can be done, break the loop
                break;
            }
        }
        return w;
    }

    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> (a[0] + a[1]) - (b[0] + b[1]));
        List<List<Integer>> result = new ArrayList<>();
        if (nums1.length == 0 || nums2.length == 0 || k == 0) {
            return result;
        }

        for (int i = 0; i < nums1.length && i < k; i++) {
            minHeap.offer(new int[]{nums1[i], nums2[0], 0});
        }
        while (k-- > 0 && !minHeap.isEmpty()) {
            int[] current = minHeap.poll();
            result.add(List.of(current[0], current[1]));
            if (current[2] + 1 < nums2.length) {
                minHeap.offer(new int[]{current[0], nums2[current[2] + 1], current[2] + 1});
            }
        }
        return result;
    }

    static class MedianFinder {
        private final PriorityQueue<Integer> maxHeap; // Max-heap for the smaller half
        private final PriorityQueue<Integer> minHeap; // Min-heap for the larger half

        public MedianFinder() {
            maxHeap = new PriorityQueue<>((a, b) -> b - a);
            minHeap = new PriorityQueue<>();
        }

        public void addNum(int num) {
            // Add to max-heap first
            maxHeap.offer(num);

            // Move the largest element from max-heap to min-heap
            minHeap.offer(maxHeap.poll());

            // Balance the heaps if necessary
            if (maxHeap.size() < minHeap.size()) {
                maxHeap.offer(minHeap.poll());
            }
        }

        public double findMedian() {
            if (maxHeap.size() > minHeap.size()) {
                // If the size of max-heap is greater, return the top of max-heap
                return maxHeap.peek();
            } else {
                // If the sizes are equal, return the average of the tops of both heaps
                return (maxHeap.peek() + minHeap.peek()) / 2.0;
            }
        }
    }

    public String addBinary(String a, String b) {
        if (a.length() > b.length()) {
            return addBinary(b, a);
        }
        String res = "";
        int carry = 0;
        for (int i = 0; i <= b.length()-1; i++) {
            int add;
            int bc = b.charAt(b.length()-1-i)-'0';
            if (i <= a.length()-1) {
                int ac = a.charAt(a.length()-1-i)-'0';
                add = ac+bc+carry;
            } else {
                add = bc+carry;
            }
            carry = add/2;
            res = String.valueOf(add%2).concat(res);
        }
        if (carry == 1) {
            res = "1".concat(res);
        }
        return res;
    }

    public int reverseBits(int n) {
        int res = 0;
        for (int i = 1; i <= 32; i++) {
            int ls = n&1;    // extract the least significant bit
            res = res<<1;
            res = res|ls;
            n = n>>1;
        }
        return res;
    }

    public int hammingWeight(int n) {
        int sb = 0;
        for (int i = 1; i <= 32; i++) {
            int ls = n&1;
            if (ls == 1) {
                sb++;
            }
            n = n>>1;
        }
        return sb;
    }

    public int singleNumber(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int res = 0;
        for (int n: nums) {
            res = res^n;    // ^: bitwise XOR, a^a=0, a^0=a, both commutative and associative
        }
        return res;
    }

    public int singleNumber2(int[] nums) {
        int p1 = 0;
        int p2 = 0;
        for (int n: nums) {
            p1 = (p1^n)&(~p2);
            p2 = (p2^n)&(~p1);
        }
        return p1;
    }

    public int rangeBitwiseAnd(int left, int right) {
        if (left == right) {
            return left;
        }
        int res = right;
        while (res > left) {
            res = res&(res-1);
        }
        return res;
    }

    public boolean isPalindrome(int x) {
        if (x < 0) {
            return false;
        }
        int origin = x;
        int rev = 0;
        while (x > 0) {
            int d = x%10;
            rev = rev*10+d;
            x /= 10;
        }
        return origin == rev;
    }

    public int[] plusOne(int[] digits) {
        ArrayList<Integer> pd = new ArrayList<>();
        int carry = 0;
        for (int i = digits.length-1; i >= 0; i--) {
            int add = digits[i];
            if (i == digits.length-1) {
                add += 1;
            }
            add += carry;
            if (add >= 10) {
                pd.addFirst(add-10);
                carry = 1;
            } else {
                pd.addFirst(add);
                carry = 0;
            }
        }
        if (carry == 1) {
            pd.addFirst(1);
        }
        int[] res = new int[pd.size()];
        for (int j = 0; j <= pd.size()-1; j++) {
            res[j] = pd.get(j);
        }
        return res;
    }
}
