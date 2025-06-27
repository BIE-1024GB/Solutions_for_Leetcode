package leetcode_75;

import java.util.*;

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
            return 1+Math.max(maxDepth(root.left), maxDepth(root.right));
        }

        private void dfs(TreeNode node, List<Integer> leaves) {
            if (node == null) {
                return;
            }
            if (node.left == null && node.right == null) {
                leaves.add(node.val);
                return;
            }
            dfs(node.left, leaves);
            dfs(node.right, leaves);
        }
        private List<Integer> findLeaf(TreeNode root) {
            List<Integer> lv = new ArrayList<>();
            dfs(root, lv);
            return lv;
        }
        public boolean leafSimilar(TreeNode root1, TreeNode root2) {
            List<Integer> v1 = findLeaf(root1);
            List<Integer> v2 = findLeaf(root2);
            if (v1.size() != v2.size()) {
                return false;
            }
            for (int i = 0; i <= v1.size()-1; i++) {
                if (!v1.get(i).equals(v2.get(i))) {
                    return false;
                }
            }
            return true;
        }

        private int GNdfs(TreeNode r, int mv) {
            if (r == null) {
                return 0;
            }
            int flag = 0;
            if (r.val >= mv) {
                flag = 1;
                flag += GNdfs(r.left, r.val);
                flag += GNdfs(r.right, r.val);
            } else {
                flag += GNdfs(r.left, mv);
                flag += GNdfs(r.right, mv);
            }
            return flag;
        }
        public int goodNodes(TreeNode root) {
            if (root.left == null && root.right == null) {
                return 1;
            }
            return 1+GNdfs(root.left, root.val)+GNdfs(root.right, root.val);
        }

        private int PSdfs(TreeNode r, int cs, int ts) {
            if (r == null) {
                return 0;
            }
            if (r.val >= 0) {
                if (cs > Integer.MAX_VALUE-r.val) {
                    return 0;
                }
            } else {
                if (cs < Integer.MIN_VALUE-r.val) {
                    return 0;
                }
            }
            int cnt = 0;
            if (r.val + cs == ts) {
                cnt += 1;
            }
            cnt = cnt+PSdfs(r.left, cs+r.val, ts)+PSdfs(r.right, cs+r.val, ts);
            return cnt;
        }
        public int pathSum(TreeNode root, int targetSum) {
            if (root == null) {
                return 0;
            }
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            ArrayList<TreeNode> list = new ArrayList<>();
            while (!queue.isEmpty()) {
                TreeNode curr = queue.poll();
                list.add(curr);
                if (curr.left != null) {
                    queue.offer(curr.left);
                }
                if (curr.right != null) {
                    queue.offer(curr.right);
                }
            }
            int ttl = 0;
            for (TreeNode tn : list) {
                ttl += PSdfs(tn, 0, targetSum);
            }
            return ttl;
        }

        private int zzl = 0;
        private void ZZdfs(TreeNode r, int dir, int cl) {
            if (r == null) {
                return;
            }
            zzl = Math.max(zzl, cl);
            if (dir == 0) {
                ZZdfs(r.right, 1, cl+1);
                ZZdfs(r.left, 0, 1);
            } else {
                ZZdfs(r.left, 0, cl+1);
                ZZdfs(r.right, 1, 1);
            }
        }
        public int longestZigZag(TreeNode root) {
            if (root.left == null && root.right == null) {
                return 0;
            }
            ZZdfs(root, 0, 0);
            ZZdfs(root, 1, 0);
            return zzl;
        }

        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            // Base case: if root is null or matches either p or q
            if (root == null || root == p || root == q) {
                return root;
            }
            // Recursively search left and right subtrees
            TreeNode left = lowestCommonAncestor(root.left, p, q);
            TreeNode right = lowestCommonAncestor(root.right, p, q);
            // If both left and right are non-null, current root is LCA
            if (left != null && right != null) {
                return root;
            }
            // Otherwise return the non-null child (if only one is found)
            return left != null ? left : right;
        }

        public List<Integer> rightSideView(TreeNode root) {
            if (root == null) {
                return new ArrayList<>();
            }
            List<Integer> res = new ArrayList<>();
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            int ls;
            while (!queue.isEmpty()) {
                ls = queue.size();
                for (int i = 1; i <= ls; i++) {
                    TreeNode curr = queue.poll();
                    assert curr != null;
                    if (curr.left != null) {
                        queue.offer(curr.left);
                    }
                    if (curr.right != null) {
                        queue.offer(curr.right);
                    }
                    if (i == ls) {
                        res.add(curr.val);
                    }
                }
            }
            return res;
        }

        public int maxLevelSum(TreeNode root) {
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            int lvl = 1;
            int ms = Integer.MIN_VALUE;
            int ml = 1;
            while (!queue.isEmpty()) {
                int ls = queue.size();
                int sum = 0;
                for (int i = 1; i <= ls; i++) {
                    TreeNode curr = queue.poll();
                    assert curr != null;
                    sum += curr.val;
                    if (curr.left != null) {
                        queue.offer(curr.left);
                    }
                    if (curr.right != null) {
                        queue.offer(curr.right);
                    }
                }
                if (sum > ms) {
                    ms = sum;
                    ml = lvl;
                }
                lvl += 1;
            }
            return ml;
        }

        public TreeNode searchBST(TreeNode root, int val) {
            TreeNode curr = root;
            while (curr != null) {
                if (val == curr.val) {
                    return curr;
                } else {
                    if (val > curr.val) {
                        curr = curr.right;
                    } else {
                        curr = curr.left;
                    }
                }
            }
            return null;
        }

        private TreeNode findMin(TreeNode node) {
            while (node.left != null) {
                node = node.left;
            }
            return node;
        }
        public TreeNode deleteNode(TreeNode root, int key) {
            if (root == null) {
                return null;
            }
            if (key < root.val) {
                root.left = deleteNode(root.left, key);
            } else if (key > root.val) {
                root.right = deleteNode(root.right, key);
            } else {
                if (root.left == null) {
                    return root.right;
                } else if (root.right == null) {
                    return root.left;
                } else {
                    TreeNode minr = findMin(root.right);
                    root.val = minr.val;
                    root.right = deleteNode(root.right, minr.val);
                }
            }
            return root;
        }
    }

    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        HashSet<Integer> set = new HashSet<>();
        set.add(0);
        Queue<Integer> queue = new LinkedList<>();
        for (int i : rooms.getFirst()) {
            queue.offer(i);
        }
        ArrayList<Integer> visit = new ArrayList<>();
        visit.add(0);
        while (!queue.isEmpty()) {
            int curr = queue.poll();
            set.add(curr);
            if (set.size() == rooms.size()) {
                return true;
            }
            if (!visit.contains(curr)) {
                for (int i : rooms.get(curr)) {
                    queue.offer(i);
                }
                visit.add(curr);
            }
        }
        return false;
    }

    private void CCdfs(int[][] ic, boolean[] v, int i) {
        for (int j = 0; j <= ic.length-1; j++) {
            if (ic[i][j] == 1 && !v[j]) {
                v[j] = true;
                CCdfs(ic, v, j);
            }
        }
    }
    public int findCircleNum(int[][] isConnected) {
        if (isConnected.length == 1) {
            return 1;
        }
        int cc = 0;
        boolean[] visit = new boolean[isConnected.length];
        for (int i = 0; i <= isConnected.length-1; i++) {
            if (!visit[i]) {
                CCdfs(isConnected, visit, i);
                cc += 1;
            }
        }
        return cc;
    }

    public int minReorder(int n, int[][] connections) {
        // Build the graph: adjacency list with direction information
        List<List<int[]>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] connection : connections) {
            int from = connection[0];
            int to = connection[1];
            // Add original edge with direction marker (1 = needs reversal)
            graph.get(from).add(new int[]{to, 1});
            // Add reverse edge with direction marker (0 = no reversal needed)
            graph.get(to).add(new int[]{from, 0});
        }

        boolean[] visited = new boolean[n];
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);
        visited[0] = true;
        int result = 0;
        while (!queue.isEmpty()) {
            int city = queue.poll();
            for (int[] neighbor : graph.get(city)) {
                int nextCity = neighbor[0];
                int direction = neighbor[1];
                if (!visited[nextCity]) {
                    visited[nextCity] = true;
                    result += direction; // only count if the edge was originally away from 0
                    queue.offer(nextCity);
                }
            }
        }
        return result;
    }

    private double dfs(String start, String end, HashMap<String, HashMap<String, Double>> graph, Set<String> visited, double product) {
        if (start.equals(end))
            return product;
        visited.add(start);
        for (Map.Entry<String, Double> entry : graph.get(start).entrySet()) {
            String next = entry.getKey();
            double value = entry.getValue();
            if (!visited.contains(next)) {
                double result = dfs(next, end, graph, visited, product * value);
                if (result != -1.0)
                    return result;
            }
        }
        return -1.0;
    }
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        HashMap<String, HashMap<String, Double>> graph = new HashMap<>();
        for (int i = 0; i < equations.size(); i++) {
            String A = equations.get(i).get(0);
            String B = equations.get(i).get(1);
            double value = values[i];
            graph.putIfAbsent(A, new HashMap<>());
            graph.putIfAbsent(B, new HashMap<>());
            graph.get(A).put(B, value);
            graph.get(B).put(A, 1.0 / value);
        }
        double[] results = new double[queries.size()];
        for (int i = 0; i < queries.size(); i++) {
            String C = queries.get(i).get(0);
            String D = queries.get(i).get(1);
            if (!graph.containsKey(C) || !graph.containsKey(D))
                results[i] = -1.0;
            else
                results[i] = dfs(C, D, graph, new HashSet<>(), 1.0);
        }
        return results;
    }

    public int nearestExit(char[][] maze, int[] entrance) {
        int m = maze.length;
        int n = maze[0].length;
        if (m == 1 && n == 1) {
            return -1;
        }
        Queue<int[]> queue = new LinkedList<>();
        HashSet<Integer> visit = new HashSet<>();
        queue.offer(entrance);
        visit.add(entrance[0]*n+entrance[1]);
        int hop = 0;
        while (!queue.isEmpty()) {
            int ls = queue.size();
            for (int i = 1; i <= ls; i++) {
                int[] curr = queue.poll();
                assert curr != null;
                if ((curr[0] == 0 || curr[0] == m - 1 || curr[1] == 0 || curr[1] == n - 1)
                        && (curr[0] != entrance[0] || curr[1] != entrance[1])) {
                    return hop;
                } else {
                    if (curr[0] >= 1 && maze[curr[0] - 1][curr[1]] != '+'
                            && !visit.contains((curr[0] - 1) * n + curr[1])) {
                        queue.offer(new int[] { curr[0] - 1, curr[1] });
                        visit.add((curr[0] - 1) * n + curr[1]);
                    }
                    if (curr[0] <= m - 2 && maze[curr[0] + 1][curr[1]] != '+'
                            && !visit.contains((curr[0] + 1) * n + curr[1])) {
                        queue.offer(new int[] { curr[0] + 1, curr[1] });
                        visit.add((curr[0] + 1) * n + curr[1]);
                    }
                    if (curr[1] >= 1 && maze[curr[0]][curr[1] - 1] != '+'
                            && !visit.contains(curr[0] * n + curr[1] - 1)) {
                        queue.offer(new int[] { curr[0], curr[1] - 1 });
                        visit.add(curr[0] * n + curr[1] - 1);
                    }
                    if (curr[1] <= n - 2 && maze[curr[0]][curr[1] + 1] != '+'
                            && !visit.contains(curr[0] * n + curr[1] + 1)) {
                        queue.offer(new int[] { curr[0], curr[1] + 1 });
                        visit.add(curr[0] * n + curr[1] + 1);
                    }
                }
            }
            hop += 1;
        }
        return -1;
    }

    public int orangesRotting(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        boolean[][] visit = new boolean[m][n];
        int fresh = 0;
        for (int i = 0; i <= m-1; i++) {
            for (int j = 0; j <= n-1; j++) {
                if (grid[i][j] == 2) {
                    queue.offer(new int[] {i, j});
                    visit[i][j] = true;
                } else if (grid[i][j] == 1) {
                    fresh += 1;
                }
            }
        }
        if (fresh == 0) {
            return 0;
        }
        int[][] dir = new int[][] {
                {-1, 0},
                {1, 0},
                {0, -1},
                {0, 1}
        };
        int minute = 1;
        while (!queue.isEmpty()) {
            int ls = queue.size();
            for (int o = 1; o <= ls; o++) {
                int[] q = queue.poll();
                for (int[] d : dir) {
                    assert q != null;
                    int nr = q[0]+d[0];
                    int nc = q[1]+d[1];
                    if (nr >= 0 && nr <= m-1 && nc >= 0 && nc <= n-1) {
                        if (grid[nr][nc] == 1 && !visit[nr][nc]) {
                            grid[nr][nc] = 2;
                            queue.offer(new int[] {nr, nc});
                            visit[nr][nc] = true;
                            fresh -= 1;
                            if (fresh == 0) {
                                return minute;
                            }
                        }
                    }
                }
            }
            minute += 1;
        }
        return -1;
    }

    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> minh = new PriorityQueue<>();
        for (int n : nums) {
            if (minh.size() < k) {
                minh.add(n);
            } else {
                if (n > minh.peek()) {
                    minh.poll();
                    minh.add(n);
                }
            }
        }
        return minh.peek();
    }

    static class SmallestInfiniteSet {
        private final PriorityQueue<Integer> addedBack;
        private final HashSet<Integer> isPresent;
        private int currentSmallest;

        public SmallestInfiniteSet() {
            addedBack = new PriorityQueue<>();
            isPresent = new HashSet<>();
            currentSmallest = 1;
        }

        public int popSmallest() {
            if (!addedBack.isEmpty()) {
                int num = addedBack.poll();
                isPresent.remove(num);
                return num;
            }
            return currentSmallest++;
        }

        public void addBack(int num) {
            // Only add back numbers that have been popped (smaller than currentSmallest) and aren't already in the set
            if (num < currentSmallest && !isPresent.contains(num)) {
                addedBack.add(num);
                isPresent.add(num);
            }
        }
    }

    public long maxScore(int[] nums1, int[] nums2, int k) {
        int[][] pair = new int[nums1.length][2];
        for (int i = 0; i <= nums1.length-1; i++) {
            pair[i][0] = nums1[i];
            pair[i][1] = nums2[i];
        }
        Arrays.sort(pair, (a, b)->b[1]-a[1]);
        PriorityQueue<Integer> minh = new PriorityQueue<>();
        long sum = 0;
        long res = 0;
        for (int[] p : pair) {
            sum += p[0];
            minh.offer(p[0]);
            if (minh.size() > k) {
                sum -= minh.poll();
            }
            if (minh.size() == k) {
                res = Math.max(res, sum*p[1]);
            }
        }
        return res;
    }

    public long totalCost(int[] costs, int k, int candidates) {
        PriorityQueue<Integer> leftHeap = new PriorityQueue<>();
        PriorityQueue<Integer> rightHeap = new PriorityQueue<>();
        int left = 0;
        int right = costs.length - 1;
        long res = 0;
        while (k > 0) {
            // Fill the left heap if there are candidates left to consider
            while (leftHeap.size() < candidates && left <= right) {
                leftHeap.offer(costs[left++]);
            }
            // Fill the right heap if there are candidates left to consider
            while (rightHeap.size() < candidates && left <= right) {
                rightHeap.offer(costs[right--]);
            }
            // Get the minimum from both heaps, treating empty heaps as having infinity
            int leftMin = leftHeap.isEmpty() ? Integer.MAX_VALUE : leftHeap.peek();
            int rightMin = rightHeap.isEmpty() ? Integer.MAX_VALUE : rightHeap.peek();
            // Choose the worker with the lowest cost
            if (leftMin <= rightMin) {
                res += leftHeap.poll();
            } else {
                res += rightHeap.poll();
            }
            k--;
        }
        return res;
    }
}
