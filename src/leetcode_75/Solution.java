package leetcode_75;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Jiarui BIE
 * @version 1.0
 * @since 2025/04/24
 */
public class Solution {
    public String mergeAlternately(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        int l = word1.length();
        int r = word2.length();
        int limit = Math.min(l, r);
        for (int i = 0; i <= limit-1; i++) {
            sb.append(word1.charAt(i));
            sb.append(word2.charAt(i));
        }
        if (limit < l) {
            for (int i = limit; i <= l-1; i++) sb.append(word1.charAt(i));
        }
        if (limit < r) {
            for (int i = limit; i <= r-1; i++) sb.append(word2.charAt(i));
        }
        return sb.toString();
    }

    private boolean isDivisor(String div, String str) {
        if (str.length()%div.length() != 0) {
            return false;
        } else {
            if (str.length() == div.length()) {
                return str.equals(div);
            } else {
                return (div.equals(str.substring(0, div.length()))) && isDivisor(div, str.substring(div.length()));
            }
        }
    }
    public String gcdOfStrings(String str1, String str2) {
        if (str1.length() > str2.length()) {
            return gcdOfStrings(str2, str1);
        } else {
            String gcd = "";
            for (int i = 1; i <= str1.length(); i++) {
                String d = str1.substring(0, i);
                if (isDivisor(d, str1) && isDivisor(d, str2)) gcd = d;
            }
            return gcd;
        }
    }

    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int mc = Integer.MIN_VALUE;
        for (int c: candies) {
            if (c > mc) {
                mc = c;
            }
        }
        List<Boolean> extra = new ArrayList<>();
        for (int i = 0; i <= candies.length-1; i++) {
            if (candies[i]+extraCandies >= mc) {
                extra.add(true);
            } else {
                extra.add(false);
            }
        }
        return extra;
    }

    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        if (n == 0) return true;
        if (flowerbed.length == 1) {
            if (flowerbed[0] == 0) {
                return 1 >= n;
            } else {
                return 0 >= n;
            }
        }
        int ap = 0;
        for (int i = 0; i <= flowerbed.length-1; i++) {
            if (flowerbed[i] == 0) {
                if (i == 0) {
                    if (flowerbed[1] == 0) {
                        ap += 1;
                        flowerbed[i] = 1;
                    }
                } else if (i == flowerbed.length-1) {
                    if (flowerbed[flowerbed.length-2] == 0) {
                        ap += 1;
                        flowerbed[i] = 1;
                    }
                } else {
                    if (flowerbed[i-1] == 0 && flowerbed[i+1] == 0) {
                        ap += 1;
                        flowerbed[i] = 1;
                    }
                }
            }
            if (ap >= n) return true;
        }
        return false;
    }

    public String reverseVowels(String s) {
        if (s.length() == 1) {
            return s;
        }
        String vowels = "AEIOUaeiou";
        int lp = 0;
        int rp = s.length()-1;
        char[] sc = s.toCharArray();
        while (lp < rp) {
            if (vowels.indexOf(s.charAt(lp)) == -1) {
                lp++;
            } else if (vowels.indexOf(s.charAt(rp)) == -1) {
                rp--;
            } else {
                char temp = sc[lp];
                sc[lp] = sc[rp];
                sc[rp] = temp;
                lp++;
                rp--;
            }
        }
        return new String(sc);
    }

    public String reverseWords(String s) {
        String[] words = s.trim().split("\\s+");
        String res = words[words.length-1];
        for (int i = words.length-2; i >= 0; i--) res = res.concat(" ").concat(words[i]);
        return res;
    }

    public int[] productExceptSelf(int[] nums) {
        int[] answer = new int[nums.length];
        int[] lv = new int[nums.length];
        int[] rv = new int[nums.length];
        for (int i = 0; i <= lv.length-1; i++) {
            if (i == 0) {
                lv[i] = 1;
            } else {
                lv[i] = nums[i-1]*lv[i-1];
            }
        }
        for (int i = rv.length-1; i >= 0; i--) {
            if (i == rv.length-1) {
                rv[i] = 1;
            } else {
                rv[i] = nums[i+1]*rv[i+1];
            }
        }
        for (int i = 0; i <= answer.length-1; i++) {
            answer[i] = lv[i]*rv[i];
        }
        return answer;
    }

    public boolean increasingTriplet(int[] nums) {
        if (nums.length < 3) return false;
        int fv = Integer.MAX_VALUE;
        int sv = Integer.MAX_VALUE;
        for (int n: nums) {
            if (n > sv) {
                return true;
            } else if (n > fv && n < sv) {
                sv = n;
            } else if (n < fv) {
                fv = n;
            }
        }
        return false;
    }

    public int compress(char[] chars) {
        if (chars.length == 1)
            return 1;
        int cl = 1;
        char cc = chars[0];
        int cp = 0;
        for (int i = 1; i <= chars.length - 1; i++) {
            if (chars[i] == cc) {
                cl += 1;
            } else {
                if (cl == 1) {
                    chars[cp] = cc;
                    cp += 1;
                } else {
                    String len = Integer.toString(cl);
                    chars[cp] = cc;
                    cp += 1;
                    for (char c : len.toCharArray()) {
                        chars[cp] = c;
                        cp += 1;
                    }
                }
                cl = 1;
                cc = chars[i];
            }
        }
        String len = Integer.toString(cl);
        chars[cp] = cc;
        cp += 1;
        if (cl > 1) {
            for (char c : len.toCharArray()) {
                chars[cp] = c;
                cp += 1;
            }
        }
        return cp;
    }

    public void moveZeroes(int[] nums) {
        // Expected approach: use 2-pointer
        if (nums.length != 1) {
            int zn = 0;
            for (int n : nums) {
                if (n == 0) {
                    zn += 1;
                }
            }
            if (zn != 0) {
                for (int i = 0; i <= nums.length-1-zn; i++) {
                    if (nums[i] == 0) {
                        int c = i;
                        for (int j = i+1; j <= nums.length-1; j++) {
                            nums[c] = nums[j];
                            nums[j] = 0;
                            c += 1;
                        }
                        i -= 1;
                    }
                }
            }
        }
    }

    public boolean isSubsequence(String s, String t) {
        if (s.isEmpty()) {
            return true;
        }
        if (t.isEmpty() || s.length() > t.length()) {
            return false;
        }
        int sp = 0;
        int tp = 0;
        while (sp <= s.length() - 1) {
            if (s.charAt(sp) == t.charAt(tp)) {
                if (sp == s.length() - 1) {
                    return true;
                } else {
                    sp++;
                    tp++;

                }
            } else {
                tp++;
            }
            if (tp > t.length() - 1) {
                return false;
            }
        }
        return false;
    }

    public int maxArea(int[] height) {
        int lb = 0;
        int rb = height.length-1;
        int water = (height.length-1)*Math.min(height[lb], height[rb]);
        while (lb < rb) {
            if (height[lb] < height[rb]) {
                lb++;
            } else {
                rb--;
            }
            int nw = (rb-lb)*Math.min(height[lb], height[rb]);
            if (nw > water) {
                water = nw;
            }
        }
        return water;
    }

    public int maxOperations(int[] nums, int k) {
        if (nums.length == 1) {
            return 0;
        }
        Arrays.sort(nums);
        int count = 0;
        int lp = 0;
        int rp = nums.length-1;
        while (lp < rp) {
            if (nums[lp]+nums[rp] == k) {
                count += 1;
                lp += 1;
                rp -= 1;
            } else {
                if (nums[lp]+nums[rp] > k) {
                    rp -= 1;
                } else {
                    lp += 1;
                }
            }
        }
        return count;
    }

    public double findMaxAverage(int[] nums, int k) {
        double mv = Double.NEGATIVE_INFINITY;
        int lp = 0;
        int rp = k-1;
        int sum = 0;
        for (int i = 0; i <= k-1; i++) {
            sum += nums[i];
        }
        while (rp <= nums.length-1) {
            double avg = ((double) sum)/((double) k);
            if (avg > mv) {
                mv = avg;
            }
            sum -= nums[lp];
            lp += 1;
            rp += 1;
            if (rp > nums.length-1) {
                break;
            }
            sum += nums[rp];
        }
        return mv;
    }

    private boolean isVowel(char c) {
        return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
    }
    public int maxVowels(String s, int k) {
        int mv = 0;
        String start = s.substring(0, k);
        int sv = 0;
        for (int i = 0; i <= start.length()-1; i++) {
            char cc = start.charAt(i);
            if (isVowel(cc)) {
                sv += 1;
            }
        }
        if (sv > mv) {
            mv = sv;
        }
        int lp = 0;
        int rp = k-1;
        while (rp < s.length()-1) {
            rp += 1;
            if (isVowel(s.charAt(rp))) {
                sv += 1;
            }
            if (isVowel(s.charAt(lp))) {
                sv -= 1;
            }
            lp += 1;
            if (sv > mv) {
                mv = sv;
            }
        }
        return mv;
    }

    public int longestOnes(int[] nums, int k) {
        int ml = -1;
        for (int i = 0; i <= nums.length-1; i++) {
            int cl = 0;
            int rf = k;
            int ci = i;
            while (ci <= nums.length-1) {
                if (nums[ci] == 1) {
                    cl += 1;
                    ci += 1;
                } else {
                    if (rf > 0) {
                        cl += 1;
                        ci += 1;
                        rf -= 1;
                    } else {
                        break;
                    }
                }
            }
            if (cl > ml) {
                ml = cl;
            }
        }
        return ml;
    }

    public int longestSubarray(int[] nums) {
        int flag = 0;
        for (int n: nums) {
            if (n == 1) {
                flag = 1;
                break;
            }
        }
        if (flag == 0) {
            return 0;
        }
        int lp = 0;
        int rp = 0;
        int d = 1;
        int ml = -1;
        while (rp <= nums.length-1) {
            if (nums[rp] == 1) {
                rp += 1;
            } else {
                if (d == 1) {
                    rp += 1;
                    d = 0;
                } else {
                    if (rp-lp-1 > ml) {
                        ml = rp-lp-1;
                    }
                    while (nums[lp] != 0) {
                        lp += 1;
                    }
                    lp += 1;
                    d = 1;
                }
            }
        }
        return Math.max(rp-lp-1, ml);
    }

    public int largestAltitude(int[] gain) {
        int ma = 0;
        int ca = 0;
        for (int g : gain) {
            ca += g;
            ma = Math.max(ca, ma);
        }
        return ma;
    }

    public int pivotIndex(int[] nums) {
        int total = 0;
        for (int n : nums) {
            total += n;
        }
        int ls = 0;
        int rs = total-nums[0];
        if (ls == rs) {
            return 0;
        }
        for (int i = 1; i <= nums.length-1; i++) {
            ls += nums[i-1];
            rs -= nums[i];
            if (ls == rs) {
                return i;
            }
        }
        return -1;
    }

    public List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> ll = new ArrayList<>();
        for (int l : nums1) {
            ll.add(l);
        }
        List<Integer> rl = new ArrayList<>();
        for (int r : nums2) {
            rl.add(r);
        }
        HashSet<Integer> sl = new HashSet<>();
        sl.addAll(ll);
        sl.addAll(rl);
        for (int n : sl) {
            if (ll.contains(n) && rl.contains(n)) {
                ll.removeIf(i -> i == n);
                rl.removeIf(i -> i == n);
            }
        }
        HashSet<Integer> ls = new HashSet<>(ll);
        ll = new ArrayList<>(ls);
        HashSet<Integer> rs = new HashSet<>(rl);
        rl = new ArrayList<>(rs);
        res.add(ll);
        res.add(rl);
        return res;
    }

    public boolean uniqueOccurrences(int[] arr) {
        if (arr.length == 1) {
            return true;
        }
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int n : arr) {
            map.put(n, map.getOrDefault(n, 0)+1);
        }
        ArrayList<Integer> occur = new ArrayList<>();
        for (int v : map.values()) {
            if (occur.contains(v)) {
                return false;
            } else {
                occur.add(v);
            }
        }
        return true;
    }

    public boolean closeStrings(String word1, String word2) {
        if (word1.length() != word2.length()) {
            return false;
        }
        HashMap<Character, Integer> m1 = new HashMap<>();
        for (int i = 0; i <= word1.length()-1; i++) {
            m1.put(word1.charAt(i), m1.getOrDefault(word1.charAt(i), 0)+1);
        }
        HashMap<Character, Integer> m2 = new HashMap<>();
        for (int i = 0; i <= word2.length()-1; i++) {
            m2.put(word2.charAt(i), m2.getOrDefault(word2.charAt(i), 0)+1);
        }
        for (char c : m1.keySet()) {
            if (!m2.containsKey(c)) {
                return false;
            }
        }
        return m1.values().stream().sorted().toList().equals(
                m2.values().stream().sorted().toList()
        );
    }

    public int equalPairs(int[][] grid) {
        int n = grid.length;
        if (n == 1) {
            return 1;
        }
        int cnt = 0;
        for (int i = 0; i <= n-1; i++) {
            ArrayList<Integer> row = new ArrayList<>();
            for (int r : grid[i]) {
                row.add(r);
            }
            for (int j = 0; j <= n-1; j++) {
                ArrayList<Integer> col = new ArrayList<>();
                for (int k = 0; k <= n-1; k++) {
                    col.add(grid[k][j]);
                }
                if (row.equals(col)) {
                    cnt += 1;
                }
            }
        }
        return cnt;
    }

    public String removeStars(String s) {
        StringBuilder stringBuffer = new StringBuilder(s);
        for (int i = 0; i <= stringBuffer.length()-1; i++) {
            char c = stringBuffer.charAt(i);
            if (c == '*') {
                stringBuffer.deleteCharAt(i);
                stringBuffer.deleteCharAt(i-1);
                i = i-2;
            }
        }
        return stringBuffer.toString();
    }

    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();
        for (int a : asteroids) {
            if (a > 0) {
                stack.push(a);
            } else {
                int size = -a;
                int flag = 1;
                while (!stack.isEmpty() && stack.peek() > 0 && size >= stack.peek()) {
                    if (size == stack.peek()) {
                        stack.pop();
                        flag = 0;
                        break;
                    } else {
                        stack.pop();
                    }
                }
                if (flag == 1) {
                    if (stack.isEmpty() || stack.peek() < 0) {
                        stack.push(a);
                    }
                }
            }
        }
        int[] array = new int[stack.size()];
        for (int i = 0; i < array.length; i++) {
            array[i] = stack.get(i);
        }
        return array;
    }

    public String decodeString(String s) {
        if (s.length() <= 3) {
            return s;
        }
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c != ']') {
                stack.push(c);
            } else {
                ArrayList<Character> list = new ArrayList<>();
                while (stack.peek() != '[') {
                    list.addFirst(stack.peek());
                    stack.pop();
                }
                stack.pop();
                int p1 = stack.peek()-'0';
                stack.pop();
                int p2 = 0;
                if (!stack.isEmpty() && Character.isDigit(stack.peek())) {
                    p2 = stack.peek()-'0';
                    stack.pop();
                }
                int p3 = 0;
                if (!stack.isEmpty() && Character.isDigit(stack.peek())) {
                    p3 = stack.peek()-'0';
                    stack.pop();
                }
                int rep = p3*100+p2*10+p1;
                while (rep > 0) {
                    for (Character cc : list) {
                        stack.push(cc);
                    }
                    rep -= 1;
                }
            }
        }
        ArrayList<Character> full = new ArrayList<>();
        while (!stack.isEmpty()) {
            full.addFirst(stack.peek());
            stack.pop();
        }
        StringBuilder builder = new StringBuilder();
        for (Character c : full) {
            builder.append(c);
        }
        return builder.toString();
    }

    static class RecentCounter {
        private final ArrayList<Integer> list;

        public RecentCounter() {
            list = new ArrayList<>();
        }

        public int ping(int t) {
            int cnt = 0;
            for (int i = list.size()-1; i >= 0; i--) {
                if (list.get(i) >= t-3000) {
                    cnt += 1;
                } else {
                    break;
                }
            }
            list.add(t);
            return cnt+1;
        }
    }

    public String predictPartyVictory(String senate) {
        if (senate.length() == 1) {
            if (senate.charAt(0) == 'R') {
                return "Radiant";
            } else {
                return "Dire";
            }
        }
        Queue<Integer> qr = new LinkedList<>();
        Queue<Integer> qd = new LinkedList<>();
        int id = 0;
        for (int i = 0; i <= senate.length()-1; i++) {
            char c = senate.charAt(i);
            if (c == 'R') {
                qr.offer(id);
            } else {
                qd.offer(id);
            }
            id += 1;
        }
        while (!qr.isEmpty() && !qd.isEmpty()) {
            int rf = qr.peek();
            int df = qd.peek();
            qd.poll();
            qr.poll();
            if (rf < df) {
                qr.offer(id);
            } else {
                qd.offer(id);
            }
            id += 1;
        }
        return qr.isEmpty() ? "Dire" : "Radiant";
    }

    static class ListNode {
        int val;
        ListNode next;
        ListNode(int val) {
            this.val = val;
        }
        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }

        public ListNode deleteMiddle(ListNode head) {
            if (head.next == null) {
                return null;
            }
            ListNode fr = head;
            int num = 0;
            while (fr != null) {
                num += 1;
                fr = fr.next;
            }
            ListNode sr = head;
            int fw = num/2-1;
            while (fw > 0) {
                sr = sr.next;
                fw -= 1;
            }
            sr.next = sr.next.next;
            return head;
        }

        public ListNode oddEvenList(ListNode head) {
            if (head == null || head.next == null || head.next.next == null) {
                return head;
            }
            ListNode oc = head;
            ListNode eh = head.next;
            ListNode ec = head.next;
            ListNode tra = head.next.next;
            int nn = 3;
            while (tra != null) {
                if (nn%2 == 0) {
                    ec.next = tra;
                    ec = ec.next;
                } else {
                    oc.next = tra;
                    oc = oc.next;
                }
                tra = tra.next;
                nn += 1;
            }
            oc.next = eh;
            ec.next = null;
            return head;
        }

        public ListNode reverseList(ListNode head) {
            // iteration approach (see LeetCode for recursion approach)
            if (head == null || head.next == null) {
                return head;
            }
            ListNode prev = head;
            ListNode curr = head.next;
            prev.next = null;
            while (curr != null) {
                ListNode temp = curr.next;
                curr.next = prev;
                prev = curr;
                curr = temp;
            }
            return prev;
        }

        public int pairSum(ListNode head) {
            if (head.next.next == null) {
                return head.val+head.next.val;
            }
            ArrayList<ListNode> nodes = new ArrayList<>();
            ListNode curr = head;
            while (curr != null) {
                nodes.add(curr);
                curr = curr.next;
            }
            int ms = Integer.MIN_VALUE;
            for (int i = 0; i <= nodes.size()/2-1; i++) {
                int cs = nodes.get(i).val+nodes.get(nodes.size()-1-i).val;
                if (cs > ms) {
                    ms = cs;
                }
            }
            return ms;
        }
    }
}
