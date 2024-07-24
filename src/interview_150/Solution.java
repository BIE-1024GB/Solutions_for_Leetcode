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
}
