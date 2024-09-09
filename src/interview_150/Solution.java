package interview_150;

import java.util.*;

public class Solution {
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
                linkedList.set(index, linkedList.get(linkedList.size()-1));
                linkedList.remove(linkedList.size()-1);
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
        StringBuffer stringBuffer = new StringBuffer();
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
        StringBuffer stringBuffer = new StringBuffer("");    // StringBuffer is thread-safe, while StringBuilder is not
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
        StringBuffer stringBuffer = new StringBuffer();
        String[] strs = s.split(" +");      // Use + for matching patterns with multiple spaces
        for (int i = strs.length-1; i >= 0; i--) {
            stringBuffer.append(strs[i]);
            if (i != 0) {
                stringBuffer.append(" ");
            }
        }
        if (stringBuffer.charAt(stringBuffer.length()-1) == ' ') {   // if the original String has space(s) at start
            stringBuffer.setLength(stringBuffer.length()-1);   // cut off the last character
        }
        return stringBuffer.toString();
    }
}
