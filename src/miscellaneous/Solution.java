package miscellaneous;

import java.util.*;

public class Solution {
    public int minimumTeachings(int n, int[][] languages, int[][] friendships) {
        // Store each user's languages in a set for quick lookup
        List<Set<Integer>> userLang = new ArrayList<>();
        for (int[] language : languages) {
            Set<Integer> set = new HashSet<>();
            for (int lang : language) {
                set.add(lang);
            }
            userLang.add(set);
        }
        // Find problematic users
        Set<Integer> problematicUsers = new HashSet<>();
        for (int[] fr : friendships) {
            int u = fr[0] - 1;
            int v = fr[1] - 1;
            // check if they share a language
            boolean ok = false;
            for (int lang : userLang.get(u)) {
                if (userLang.get(v).contains(lang)) {
                    ok = true;
                    break;
                }
            }
            if (!ok) {
                problematicUsers.add(u);
                problematicUsers.add(v);
            }
        }
        // If no problematic friendships, no need to teach
        if (problematicUsers.isEmpty()) return 0;
        int res = Integer.MAX_VALUE;
        // Try each language
        for (int lang = 1; lang <= n; lang++) {
            int count = 0;
            for (int u : problematicUsers) {
                if (userLang.get(u).contains(lang)) {
                    count++;
                }
            }
            res = Math.min(res, problematicUsers.size() - count);
        }
        return res;
    }

    private int gcd(int a, int b) {
        // Efficient Euclidean method
        while (b != 0) {
            int tmp = a % b;
            a = b;
            b = tmp;
        }
        return a;
    }
    private int lcm(int a, int b) {
        return (int) ((long) a / gcd(a, b) * b);
    }
    public List<Integer> replaceNonCoprimes(int[] nums) {
        Deque<Integer> stack = new ArrayDeque<>();  // ArrayDeque is preferable over old Stack
        for (int n : nums) {
            stack.addLast(n);
            // Keep merging with previous while non-coprime
            while (stack.size() > 1) {
                int b = stack.removeLast();
                int a = stack.removeLast();
                int g = gcd(a, b);
                if (g > 1) {
                    // Merge into LCM and continue checking
                    int l = lcm(a, b);
                    stack.addLast(l);
                } else {
                    // Put back and stop merging
                    stack.addLast(a);
                    stack.addLast(b);
                    break;
                }
            }
        }
        return new ArrayList<>(stack);
    }

    static class FoodRatings {
        private final Map<String, String> foodToCuisine;
        private final Map<String, Integer> foodToRating;
        private final Map<String, PriorityQueue<Food>> cuisineToPQ;

        public FoodRatings(String[] foods, String[] cuisines, int[] ratings) {
            foodToCuisine = new HashMap<>();
            foodToRating = new HashMap<>();
            cuisineToPQ = new HashMap<>();

            for (int i = 0; i < foods.length; i++) {
                String food = foods[i];
                String cuisine = cuisines[i];
                int rating = ratings[i];
                foodToCuisine.put(food, cuisine);
                foodToRating.put(food, rating);
                cuisineToPQ.computeIfAbsent(cuisine, k -> new PriorityQueue<>()).add(new Food(food, rating));
            }
        }

        public void changeRating(String food, int newRating) {
            foodToRating.put(food, newRating);
            String cuisine = foodToCuisine.get(food);
            cuisineToPQ.get(cuisine).add(new Food(food, newRating));
        }

        public String highestRated(String cuisine) {
            PriorityQueue<Food> pq = cuisineToPQ.get(cuisine);
            // Lazy cleanup of outdated entries
            while (true) {
                Food top = pq.peek();
                assert top != null;
                if (foodToRating.get(top.name) == top.rating) {
                    return top.name;
                }
                pq.poll(); // remove stale entry
            }
        }

        private static class Food implements Comparable<Food> {
            String name;
            int rating;

            Food(String n, int r) {
                name = n;
                rating = r;
            }

            @Override
            public int compareTo(Food other) {
                if (this.rating != other.rating) {
                    return other.rating - this.rating; // higher rating first
                }
                return this.name.compareTo(other.name); // lexicographically smaller first
            }
        }
    }

    static class TaskManager {
        private static class Task {
            int tid;
            int priority;

            public Task(int tid, int priority) {
                this.tid = tid;
                this.priority = priority;
            }
        }

        private final HashMap<Integer, Integer> tidTOuid;
        private final HashMap<Integer, Integer> tidTOpri;
        private final PriorityQueue<Task> pq;

        public TaskManager(List<List<Integer>> tasks) {
            tidTOuid = new HashMap<>();
            tidTOpri = new HashMap<>();
            pq = new PriorityQueue<>((a, b)->{
                if (a.priority != b.priority) {
                    return Integer.compare(b.priority, a.priority);
                }
                return Integer.compare(b.tid, a.tid);
            });
            for (List<Integer> t : tasks) {
                int uid = t.get(0);
                int tid = t.get(1);
                int pri = t.get(2);
                tidTOuid.put(tid, uid);
                tidTOpri.put(tid, pri);
                pq.add(new Task(tid, pri));
            }
        }

        public void add(int userId, int taskId, int priority) {
            tidTOuid.put(taskId, userId);
            tidTOpri.put(taskId, priority);
            pq.add(new Task(taskId, priority));
        }

        public void edit(int taskId, int newPriority) {
            tidTOpri.put(taskId, newPriority);
            pq.add(new Task(taskId, newPriority));
        }

        public void rmv(int taskId) {
            tidTOuid.remove(taskId);
            tidTOpri.remove(taskId);
        }

        public int execTop() {
            if (tidTOuid.isEmpty()) {
                return -1;
            } else {
                while (true) {
                    Task curr = pq.peek();
                    assert curr != null;
                    if (tidTOuid.containsKey(curr.tid)) {
                        if (tidTOpri.get(curr.tid) == curr.priority) {
                            pq.poll();
                            int res = tidTOuid.get(curr.tid);
                            rmv(curr.tid);
                            return res;
                        }
                    }
                    pq.poll();
                }
            }
        }
    }

    static class Spreadsheet {
        private final int[][] sheet;
        public Spreadsheet(int rows) {
            sheet = new int[rows][26];
        }

        public void setCell(String cell, int value) {
            int col = cell.charAt(0)-'A';
            int row = Integer.parseInt(cell.substring(1))-1;
            sheet[row][col] = value;
        }

        public void resetCell(String cell) {
            int col = cell.charAt(0)-'A';
            int row = Integer.parseInt(cell.substring(1))-1;
            sheet[row][col] = 0;
        }

        public int getValue(String formula) {
            String body = formula.substring(1);
            String[] parts = body.split("\\+");  // need to escape '+'
            int fv;
            if (Character.isLetter(parts[0].charAt(0))) {
                int col = parts[0].charAt(0)-'A';
                int row = Integer.parseInt(parts[0].substring(1))-1;
                fv = sheet[row][col];
            } else {
                fv = Integer.parseInt(parts[0]);
            }
            int sv;
            if (Character.isLetter(parts[1].charAt(0))) {
                int col = parts[1].charAt(0)-'A';
                int row = Integer.parseInt(parts[1].substring(1))-1;
                sv = sheet[row][col];
            } else {
                sv = Integer.parseInt(parts[1]);
            }
            return fv+sv;
        }
    }

    static class Router {
        private static class Packet {
            int source, destination, timestamp;

            Packet(int s, int d, int t) {
                this.source = s;
                this.destination = d;
                this.timestamp = t;
            }
        }

        private final int memoryLimit;
        private final Queue<Packet> queue; // FIFO order
        private final Set<String> seen; // to detect duplicates
        private final Map<Integer, TreeMap<Integer, Integer>> destMap;
        // destination -> (timestamp -> count of packets with this timestamp)

        public Router(int memoryLimit) {
            this.memoryLimit = memoryLimit;
            this.queue = new ArrayDeque<>();
            this.seen = new HashSet<>();
            this.destMap = new HashMap<>();
        }

        private String key(int source, int destination, int timestamp) {
            return source + "#" + destination + "#" + timestamp;
        }

        public boolean addPacket(int source, int destination, int timestamp) {
            String k = key(source, destination, timestamp);
            if (seen.contains(k))
                return false; // duplicate

            // If memory full, evict oldest packet
            if (queue.size() == memoryLimit) {
                Packet old = queue.poll();
                assert old != null;
                seen.remove(key(old.source, old.destination, old.timestamp));

                TreeMap<Integer, Integer> tm = destMap.get(old.destination);
                tm.put(old.timestamp, tm.get(old.timestamp) - 1);
                if (tm.get(old.timestamp) == 0) {
                    tm.remove(old.timestamp);
                }
                if (tm.isEmpty())
                    destMap.remove(old.destination);
            }

            Packet p = new Packet(source, destination, timestamp);
            queue.offer(p);
            seen.add(k);

            destMap.computeIfAbsent(destination, x -> new TreeMap<>())
                    .merge(timestamp, 1, Integer::sum);

            return true;
        }

        public int[] forwardPacket() {
            if (queue.isEmpty())
                return new int[0];
            Packet p = queue.poll();
            seen.remove(key(p.source, p.destination, p.timestamp));

            TreeMap<Integer, Integer> tm = destMap.get(p.destination);
            tm.put(p.timestamp, tm.get(p.timestamp) - 1);
            if (tm.get(p.timestamp) == 0) {
                tm.remove(p.timestamp);
            }
            if (tm.isEmpty())
                destMap.remove(p.destination);

            return new int[] { p.source, p.destination, p.timestamp };
        }

        public int getCount(int destination, int startTime, int endTime) {
            if (!destMap.containsKey(destination))
                return 0;
            TreeMap<Integer, Integer> tm = destMap.get(destination);

            // Get submap of timestamps in range [startTime, endTime]
            NavigableMap<Integer, Integer> sub = tm.subMap(startTime, true, endTime, true);
            int count = 0;
            for (int v : sub.values())
                count += v;
            return count;
        }
    }

    public int reverse(int x) {
        int res = 0;
        while (x != 0) {
            int digit = x%10;
            if (res>Integer.MAX_VALUE/10 || (res==Integer.MAX_VALUE/10&&digit>7)) {
                return 0;
            } else if (res<Integer.MIN_VALUE/10 || (res==Integer.MIN_VALUE/10&&digit<-8)) {
                return 0;
            } else {
                res = res*10+digit;
                x /= 10;
            }
        }
        return res;
    }

    public int compareVersion(String version1, String version2) {
        int d1 = 0;
        int d2 = 0;
        for (char c : version1.toCharArray()) {
            if (c == '.') {
                d1 += 1;
            }
        }
        for (char c : version2.toCharArray()) {
            if (c == '.') {
                d2 += 1;
            }
        }
        if (d1 != d2) {
            if (d1 < d2) {
                int diff = d2-d1;
                while (diff > 0) {
                    version1 = version1.concat(".0");
                    diff--;
                }
            } else {
                int diff = d1-d2;
                while (diff > 0) {
                    version2 = version2.concat(".0");
                    diff--;
                }
            }
            return compareVersion(version1, version2);
        } else {
            String[] p1 = version1.split("\\.");
            String[] p2 = version2.split("\\.");    // '.' is a meta character
            for (int i = 0; i <= p1.length-1; i++) {
                int n1 = Integer.parseInt(p1[i]);
                int n2 = Integer.parseInt(p2[i]);
                if (n1 < n2) {
                    return -1;
                } else if (n1 > n2) {
                    return 1;
                }
            }
            return 0;
        }
    }

    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0)
            return "0";

        StringBuilder sb = new StringBuilder();

        // Handle sign
        if ((numerator < 0) ^ (denominator < 0)) {
            sb.append("-");
        }

        // Convert to long to avoid overflow (like -2147483648 case)
        long num = Math.abs((long) numerator);
        long den = Math.abs((long) denominator);

        // Integer part
        sb.append(num / den);
        long remainder = num % den;
        if (remainder == 0) {
            return sb.toString();
        }

        sb.append(".");

        // Map remainder -> position in StringBuilder
        Map<Long, Integer> remainderPos = new HashMap<>();
        while (remainder != 0) {
            if (remainderPos.containsKey(remainder)) {
                int start = remainderPos.get(remainder);
                sb.insert(start, "(");
                sb.append(")");
                break;
            }

            remainderPos.put(remainder, sb.length());
            remainder *= 10;
            sb.append(remainder / den);
            remainder %= den;
        }

        return sb.toString();
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle.size() == 1) {
            return triangle.getFirst().getFirst();
        }
        int n = triangle.size();
        int[][] dp = new int[n][n];
        dp[0][0] = triangle.getFirst().getFirst();
        int res = Integer.MAX_VALUE;
        for (int i = 1; i <= n-1; i++) {
            for (int j = 0; j <= i; j++) {
                if (j == 0) {
                    dp[i][j] = dp[i-1][j]+triangle.get(i).get(j);
                } else if (j == i) {
                    dp[i][j] = dp[i-1][j-1]+triangle.get(i).get(j);
                } else {
                    dp[i][j] = Math.min(dp[i-1][j-1], dp[i-1][j])+triangle.get(i).get(j);
                }
                if (i == n-1) {
                    if (dp[i][j] < res) {
                        res = dp[i][j];
                    }
                }
            }
        }
        return res;
    }

    public int triangleNumber(int[] nums) {
        if (nums.length <= 2) {
            return 0;
        }
        Arrays.sort(nums);
        int n = nums.length;
        int count = 0;
        for (int k = n - 1; k >= 2; k--) {
            int i = 0, j = k - 1;
            while (i < j) {
                if (nums[i] + nums[j] > nums[k]) {
                    count += (j - i);
                    j--;
                } else {
                    i++;
                }
            }
        }
        return count;
    }

    public double largestTriangleArea(int[][] points) {
        double ba = Double.MIN_VALUE;
        for (int i = 0; i <= points.length-3; i++) {
            for (int j = i+1; j <= points.length-2; j++) {
                for (int k = j+1; k <= points.length-1; k++) {
                    int[] p1 = points[i];
                    int[] p2 = points[j];
                    int[] p3 = points[k];
                    double ca = (double) Math.abs(p1[0]*(p2[1]-p3[1])+p2[0]*(p3[1]-p1[1])+p3[0]*(p1[1]-p2[1]))/2;
                    if (ca > ba) {
                        ba = ca;
                    }
                }
            }
        }
        return ba;
    }

    public int largestPerimeter(int[] nums) {
        Arrays.sort(nums);
        for (int i = nums.length-1; i >= 2; i--) {
            if (nums[i-2]+nums[i-1] > nums[i]) {
                return nums[i-2]+nums[i-1]+nums[i];
            }
        }
        return 0;
    }

    public int myAtoi(String s) {
        if (s.isEmpty()) {
            return 0;
        }
        int i = 0, n = s.length();
        // 1. Skip leading whitespaces
        while (i < n && s.charAt(i) == ' ') {
            i++;
        }
        if (i == n) {
            return 0;
        }
        // 2. Handle sign
        int sign = 1;
        if (s.charAt(i) == '+' || s.charAt(i) == '-') {
            sign = (s.charAt(i) == '-') ? -1 : 1;
            i++;
        }
        // 3. Convert digits and check overflow
        long result = 0;
        while (i < n && Character.isDigit(s.charAt(i))) {
            int digit = s.charAt(i) - '0';
            result = result * 10 + digit;
            // clamp when overflow
            if (sign == 1 && result > Integer.MAX_VALUE) {
                return Integer.MAX_VALUE;
            }
            if (sign == -1 && -result < Integer.MIN_VALUE) {
                return Integer.MIN_VALUE;
            }
            i++;
        }
        return (int) (sign * result);
    }

    public int minScoreTriangulation(int[] values) {
        int n = values.length;
        int[][] dp = new int[n][n];
        // length is the distance between i and j
        for (int len = 2; len < n; len++) {
            for (int i = 0; i + len < n; i++) {
                int j = i + len;
                dp[i][j] = Integer.MAX_VALUE;
                for (int k = i + 1; k < j; k++) {
                    int cost = dp[i][k] + dp[k][j] + values[i] * values[j] * values[k];
                    dp[i][j] = Math.min(dp[i][j], cost);
                }
            }
        }
        return dp[0][n - 1];
    }

    public int triangularSum(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int[] prev = nums;
        while (prev.length > 1) {
            int[] nn = new int[prev.length-1];
            for (int i = 0; i <= nn.length-1; i++) {
                nn[i] = (prev[i]+prev[i+1])%10;
            }
            prev = nn;
        }
        return prev[0];
    }

    public int numWaterBottles(int numBottles, int numExchange) {
        int res = 0;
        int empt = 0;
        int fb = numBottles;
        while (fb+empt >= numExchange) {
            res += fb;
            empt += fb;
            fb = empt/numExchange;
            empt %= numExchange;
        }
        res += fb;
        return res;
    }

    public int maxBottlesDrunk(int numBottles, int numExchange) {
        int res = 0;
        int empt = 0;
        while (numBottles>0 || empt>=numExchange) {
            if (numBottles > 0) {
                res += numBottles;
                empt += numBottles;
                numBottles = 0;
            } else {
                numBottles = 1;
                empt -= numExchange;
                numExchange += 1;
            }
        }
        return res;
    }

    static class Cell {
        int i, j, height;

        Cell(int i, int j, int height) {
            this.i = i;
            this.j = j;
            this.height = height;
        }
    }
    public int trapRainWater(int[][] heightMap) {
        int m = heightMap.length, n = heightMap[0].length;
        if (m <= 2 || n <= 2)
            return 0;

        // Min-heap (lowest height first)
        PriorityQueue<Cell> pq = new PriorityQueue<>((a, b) -> a.height - b.height);
        boolean[][] visited = new boolean[m][n];

        // Add all boundary cells into the heap
        for (int i = 0; i < m; i++) {
            pq.offer(new Cell(i, 0, heightMap[i][0]));
            pq.offer(new Cell(i, n - 1, heightMap[i][n - 1]));
            visited[i][0] = true;
            visited[i][n - 1] = true;
        }
        for (int j = 1; j < n - 1; j++) {
            pq.offer(new Cell(0, j, heightMap[0][j]));
            pq.offer(new Cell(m - 1, j, heightMap[m - 1][j]));
            visited[0][j] = true;
            visited[m - 1][j] = true;
        }

        int totalWater = 0;
        int[][] dirs = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };

        while (!pq.isEmpty()) {
            Cell cell = pq.poll();

            for (int[] d : dirs) {
                int ni = cell.i + d[0];
                int nj = cell.j + d[1];
                if (ni < 0 || nj < 0 || ni >= m || nj >= n || visited[ni][nj])
                    continue;

                visited[ni][nj] = true;
                int neighborHeight = heightMap[ni][nj];
                // If neighbor is lower, water can be trapped
                if (neighborHeight < cell.height) {
                    totalWater += cell.height - neighborHeight;
                }
                // Push the neighbor with the effective boundary height
                pq.offer(new Cell(ni, nj, Math.max(neighborHeight, cell.height)));
            }
        }

        return totalWater;
    }

    public int maxArea(int[] height) {
        int lp = 0;
        int rp = height.length-1;
        int mw = 0;
        while (lp < rp) {
            int cw = Math.min(height[lp], height[rp])*(rp-lp);
            mw = Math.max(mw, cw);
            if (height[lp] <= height[rp]) {
                lp++;
            } else {
                rp--;
            }
        }
        return mw;
    }

    public int subarraySum(int[] nums, int k) {
        HashMap<Integer, Integer> pf = new HashMap<>();
        pf.put(0, 1);
        int ps = 0;
        int res = 0;
        for (int n : nums) {
            ps += n;
            res += pf.getOrDefault(ps-k, 0);
            pf.put(ps, pf.getOrDefault(ps, 0)+1);
        }
        return res;
    }

    public int[] avoidFlood(int[] rains) {
        int n = rains.length;
        int[] ans = new int[n];
        Map<Integer, Integer> fullLakes = new HashMap<>();
        TreeSet<Integer> dryDays = new TreeSet<>();
        for (int i = 0; i < n; i++) {
            int lake = rains[i];
            if (lake == 0) {
                dryDays.add(i);
                ans[i] = 1;
            } else {
                ans[i] = -1;
                if (fullLakes.containsKey(lake)) {
                    // Find the next dry day after the last rain on this lake
                    Integer dryDay = dryDays.higher(fullLakes.get(lake));
                    if (dryDay == null) {
                        return new int[0]; // impossible to prevent flood
                    }
                    ans[dryDay] = lake; // dry this lake on that day
                    dryDays.remove(dryDay);
                }
                fullLakes.put(lake, i); // mark the lake as full (last filled today)
            }
        }
        return ans;
    }

    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        Arrays.sort(potions);
        int[] res = new int[spells.length];
        for (int i = 0; i <= res.length - 1; i++) {
            long st = (success + spells[i] - 1) / spells[i];
            int lp = 0;
            int rp = potions.length;
            while (lp < rp) {
                int mid = lp + (rp - lp) / 2;
                if (potions[mid] < st) {
                    lp = mid + 1;
                } else {
                    rp = mid;
                }
            }
            res[i] = potions.length - lp;
        }
        return res;
    }

    public long minTime(int[] skill, int[] mana) {
        int n = skill.length, m = mana.length;
        // prefix for previous job (size n)
        long[] prevPrefix = new long[n];

        // prefix for job 0
        prevPrefix[0] = (long) skill[0] * mana[0];
        for (int i = 1; i < n; i++) {
            prevPrefix[i] = prevPrefix[i - 1] + (long) skill[i] * mana[0];
        }

        long start = 0L; // S_0

        // process jobs 1..m-1
        for (int j = 1; j < m; j++) {
            long[] currPrefix = new long[n];
            currPrefix[0] = (long) skill[0] * mana[j];
            for (int i = 1; i < n; i++) {
                currPrefix[i] = currPrefix[i - 1] + (long) skill[i] * mana[j];
            }

            // compute D = max_i (prevPrefix[i] - currPrefix[i-1]) with currPrefix[-1] = 0
            long D = prevPrefix[0]; // i = 0 case: prevPrefix[0] - 0
            for (int i = 1; i < n; i++) {
                long val = prevPrefix[i] - currPrefix[i - 1];
                if (val > D)
                    D = val;
            }
            if (D < 0)
                D = 0;
            start += D;

            prevPrefix = currPrefix;
        }

        // makespan = start of last job + total time of last job
        return start + prevPrefix[n - 1];
    }

    static class LRUCache {
        static class Node {
            Node prev;
            Node next;
            int key;
            int val;

            public Node(int k, int v) {
                key = k;
                val = v;
            }
        }

        private final int capacity;
        private final HashMap<Integer, Node> map;
        private final Node head;
        private final Node tail;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            map = new HashMap<>();
            head = new Node(-1, -1);
            tail = new Node(-1, -1);
            head.next = tail;
            tail.prev = head;
        }

        private void pop(Node n) {
            Node prev = n.prev;
            Node next = n.next;
            prev.next = next;
            next.prev = prev;
        }

        private void push(Node n) {
            Node hn = head.next;
            head.next = n;
            n.next = hn;
            n.prev = head;
            hn.prev = n;
        }

        public int get(int key) {
            if (map.containsKey(key)) {
                Node target = map.get(key);
                pop(target);
                push(target);
                return target.val;
            } else {
                return -1;
            }
        }

        public void put(int key, int value) {
            if (map.containsKey(key)) {
                Node target = map.get(key);
                target.val = value;
                pop(target);
                push(target);
            } else {
                Node nn = new Node(key, value);
                push(nn);
                map.put(key, nn);
                if (map.size() > capacity) {
                    Node dump = tail.prev;
                    pop(dump);
                    map.remove(dump.key);
                }
            }
        }
    }

    public int maximumEnergy(int[] energy, int k) {
        int[] dp = new int[energy.length];
        if (k - 1 + 1 >= 0) System.arraycopy(energy, 0, dp, 0, k - 1 + 1);
        for (int i = k; i <= dp.length-1; i++) {
            dp[i] = Math.max(energy[i], energy[i]+dp[i-k]);
        }
        int me = Integer.MIN_VALUE;
        for (int i = dp.length-k; i <= dp.length-1; i++) {
            me = Math.max(me, dp[i]);
        }
        return me;
    }

    private int binarySearch(List<Integer> vals, int i) {
        // Find largest index j < i where vals[i] - vals[j] > 2
        int lo = 0, hi = i - 1, ans = -1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (vals.get(i) - vals.get(mid) > 2) {
                ans = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return ans;
    }
    public long maximumTotalDamage(int[] power) {
        // Step 1: Count total damage per unique power
        Map<Integer, Long> map = new HashMap<>();
        for (int p : power) {
            map.put(p, map.getOrDefault(p, 0L) + p);
        }

        // Step 2: Sort unique power values
        List<Integer> vals = new ArrayList<>(map.keySet());
        Collections.sort(vals);
        int n = vals.size();

        // Step 3: Prepare DP array
        long[] dp = new long[n];
        dp[0] = map.get(vals.getFirst());

        for (int i = 1; i < n; i++) {
            long take = map.get(vals.get(i));

            // Binary search for last non-conflicting index j
            int j = binarySearch(vals, i);
            if (j != -1)
                take += dp[j];

            dp[i] = Math.max(dp[i - 1], take);
        }

        return dp[n - 1];
    }

    public List<String> removeAnagrams(String[] words) {
        List<String> res = new ArrayList<>();
        if (words.length == 1) {
            res.add(words[0]);
            return res;
        }
        res.addAll(Arrays.asList(words));
        while (res.size()>=2) {
            boolean mod = false;
            for (int i = 1; i <= res.size()-1; i++) {
                String curr = res.get(i);
                HashMap<Character, Integer> map = new HashMap<>();
                for (char c : curr.toCharArray()) {
                    map.put(c, map.getOrDefault(c, 0)+1);
                }
                String prev = res.get(i-1);
                HashMap<Character, Integer> pm = new HashMap<>();
                for (char c : prev.toCharArray()) {
                    pm.put(c, pm.getOrDefault(c, 0)+1);
                }
                boolean ana = true;
                if (map.size() == pm.size()) {
                    for (Character c : map.keySet()) {
                        if (!pm.containsKey(c) || !Objects.equals(map.get(c), pm.get(c))) {
                            ana = false;
                            break;
                        }
                    }
                } else {
                    ana = false;
                }
                if (ana) {
                    res.remove(i);
                    mod = true;
                    break;
                }
            }
            if (!mod) {
                break;
            }
        }
        return res;
    }

    public boolean hasIncreasingSubarrays(List<Integer> nums, int k) {
        int i = 0;
        int len = 0;
        int fp = Integer.MIN_VALUE;
        int sp = Integer.MIN_VALUE;
        while (i <= nums.size()-k-1) {
            if (nums.get(i) > fp && nums.get(i+k) > sp) {
                len += 1;
                if (len == k) {
                    return true;
                }
            } else {
                len = 1;
            }
            fp = nums.get(i);
            sp = nums.get(i+k);
            i++;
        }
        return false;
    }

    public int maxIncreasingSubarrays(List<Integer> nums) {
        int n = nums.size();
        int[] inc = new int[n];
        int[] incEnd = new int[n];

        // Compute increasing lengths starting at each index
        inc[n - 1] = 1;
        for (int i = n - 2; i >= 0; i--) {
            if (nums.get(i) < nums.get(i + 1)) {
                inc[i] = inc[i + 1] + 1;
            } else {
                inc[i] = 1;
            }
        }
        // Compute increasing lengths ending at each index
        incEnd[0] = 1;
        for (int i = 1; i < n; i++) {
            if (nums.get(i - 1) < nums.get(i)) {
                incEnd[i] = incEnd[i - 1] + 1;
            } else {
                incEnd[i] = 1;
            }
        }

        // Find maximum k where both adjacent segments of length k are strictly increasing
        int maxK = 1;
        for (int i = 0; i < n - 1; i++) {
            int k = Math.min(incEnd[i], inc[i + 1]);
            if (k > maxK)
                maxK = k;
        }

        return maxK;
    }

    static class Bank {
        private final long[] balance;

        public Bank(long[] balance) {
            this.balance = balance.clone();
        }

        public boolean transfer(int account1, int account2, long money) {
            if (account1<1 || account1>balance.length || account2<1 || account2>balance.length) {
                return false;
            }
            long cm = balance[account1-1];
            if (money > cm) {
                return false;
            }
            balance[account1-1]-=money;
            balance[account2-1]+=money;
            return true;
        }

        public boolean deposit(int account, long money) {
            if (account<1 || account>balance.length) {
                return false;
            }
            balance[account-1]+=money;
            return true;
        }

        public boolean withdraw(int account, long money) {
            if (account<1 || account>balance.length) {
                return false;
            }
            long cm = balance[account-1];
            if (cm < money) {
                return false;
            }
            balance[account-1]-=money;
            return true;
        }
    }

    public int numberOfBeams(String[] bank) {
        if (bank.length == 1) {
            return 0;
        }
        boolean hasDevice = false;
        int fr = 0;
        for (int i = 0; i <= bank.length-1; i++) {
            String b = bank[i];
            if (b.contains("1")) {
                hasDevice = true;
                fr = i;
                break;
            }
        }
        if (!hasDevice || fr==bank.length-1) {
            return 0;
        }
        int res = 0;
        int cd = 0;
        for (char c : bank[fr].toCharArray()) {
            if (c == '1') {
                cd += 1;
            }
        }
        for (int j = fr+1; j <= bank.length-1; j++) {
            String curr = bank[j];
            if (curr.contains("1")) {
                int nd = 0;
                for (char c : curr.toCharArray()) {
                    if (c == '1') {
                        nd += 1;
                    }
                }
                res += nd*cd;
                cd = nd;
            }
        }
        return res;
    }

    public int countValidSelections(int[] nums) {
        if (nums.length == 1) {
            return 2;
        }
        int[] prefix = new int[nums.length];
        int[] suffix = new int[nums.length];
        for (int i = 1; i <= nums.length-1; i++) {
            prefix[i] = prefix[i-1]+nums[i-1];
            suffix[nums.length-1-i] = suffix[nums.length-i]+nums[nums.length-i];
        }
        int res = 0;
        for (int i = 0; i <= nums.length-1; i++) {
            if (nums[i] == 0) {
                if (prefix[i] == suffix[i]) {
                    res += 2;
                } else if (Math.abs(prefix[i]-suffix[i]) == 1) {
                    res += 1;
                }
            }
        }
        return res;
    }

    public int smallestNumber(int n) {
        int res = 2;
        while (res-1 < n) {
            res *= 2;
        }
        return res-1;
    }

    public int minNumberOperations(int[] target) {
        int op = target[0];
        for (int i = 1; i <= target.length-1; i++) {
            if (target[i] > target[i-1]) {
                op += (target[i]-target[i-1]);
            }
        }
        return op;
    }

    public int[] getSneakyNumbers(int[] nums) {
        int[] res = new int[2];
        HashSet<Integer> set = new HashSet<>();
        int idx = 0;
        for (int n : nums) {
            if (!set.contains(n)) {
                set.add(n);
            } else {
                res[idx] = n;
                idx++;
                if (idx == 2) {
                    break;
                }
            }
        }
        return res;
    }

    public int minCost(String colors, int[] neededTime) {
        if (colors.length() == 1) {
            return 0;
        }
        int res = 0;
        char curr = colors.charAt(0);
        int ci = 0;
        for (int i = 1; i <= colors.length()-1; i++) {
            char n = colors.charAt(i);
            if (n == curr) {
                int sum = neededTime[ci]+neededTime[i];
                int mt = Math.max(neededTime[ci], neededTime[i]);
                int j = i+1;
                while (j <= colors.length()-1 && colors.charAt(j) == curr) {
                    sum += neededTime[j];
                    mt = Math.max(mt, neededTime[j]);
                    j++;
                }
                res += (sum-mt);
                i = j;

            }
            if (i >= colors.length()) {
                break;
            }
            curr = colors.charAt(i);
            ci = i;
        }
        return res;
    }

    public int[] findXSum(int[] nums, int k, int x) {
        int[] res = new int[nums.length-k+1];
        for (int i = 0; i <= nums.length-k; i++) {
            HashMap<Integer, Integer> map = new HashMap<>();
            for (int j = i; j <= i+k-1; j++) {
                map.put(nums[j], map.getOrDefault(nums[j], 0)+1);
            }
            ArrayList<Map.Entry<Integer, Integer>> list = new ArrayList<>(map.entrySet());
            list.sort((a, b)->{
                if (!a.getValue().equals(b.getValue())) {
                    return b.getValue()-a.getValue();
                } else {
                    return b.getKey()-a.getKey();
                }
            });
            int sum = 0;
            for (int l = 0; l <= Math.min(list.size(), x)-1; l++) {
                Map.Entry<Integer, Integer> e = list.get(l);
                sum += e.getKey()*e.getValue();
            }
            res[i] = sum;
        }
        return res;
    }

    static class Helper {

        private final int x;
        private long result;
        private final TreeSet<Pair> large;
        private final TreeSet<Pair> small;
        private final Map<Integer, Integer> occ;

        private static class Pair implements Comparable<Pair> {

            int first;
            int second;

            Pair(int first, int second) {
                this.first = first;
                this.second = second;
            }

            @Override
            public int compareTo(Pair other) {
                if (this.first != other.first) {
                    return Integer.compare(this.first, other.first);
                }
                return Integer.compare(this.second, other.second);
            }

            @Override
            public boolean equals(Object obj) {
                if (this == obj)
                    return true;
                if (obj == null || getClass() != obj.getClass())
                    return false;
                Pair pair = (Pair) obj;
                return first == pair.first && second == pair.second;
            }

            @Override
            public int hashCode() {
                return Objects.hash(first, second);
            }
        }

        public Helper(int x) {
            this.x = x;
            this.result = 0;
            this.large = new TreeSet<>();
            this.small = new TreeSet<>();
            this.occ = new HashMap<>();
        }

        public void insert(int num) {
            if (occ.containsKey(num) && occ.get(num) > 0) {
                internalRemove(new Pair(occ.get(num), num));
            }
            occ.put(num, occ.getOrDefault(num, 0) + 1);
            internalInsert(new Pair(occ.get(num), num));
        }

        public void remove(int num) {
            internalRemove(new Pair(occ.get(num), num));
            occ.put(num, occ.get(num) - 1);
            if (occ.get(num) > 0) {
                internalInsert(new Pair(occ.get(num), num));
            }
        }

        public long get() {
            return result;
        }

        private void internalInsert(Pair p) {
            if (large.size() < x || p.compareTo(large.first()) > 0) {
                result += (long) p.first * p.second;
                large.add(p);
                if (large.size() > x) {
                    Pair toRemove = large.first();
                    result -= (long) toRemove.first * toRemove.second;
                    large.remove(toRemove);
                    small.add(toRemove);
                }
            } else {
                small.add(p);
            }
        }

        private void internalRemove(Pair p) {
            if (p.compareTo(large.first()) >= 0) {
                result -= (long) p.first * p.second;
                large.remove(p);
                if (!small.isEmpty()) {
                    Pair toAdd = small.last();
                    result += (long) toAdd.first * toAdd.second;
                    small.remove(toAdd);
                    large.add(toAdd);
                }
            } else {
                small.remove(p);
            }
        }
    }
    public long[] findXSumII(int[] nums, int k, int x) {
        Helper helper = new Helper(x);
        List<Long> ans = new ArrayList<>();

        for (int i = 0; i < nums.length; i++) {
            helper.insert(nums[i]);
            if (i >= k) {
                helper.remove(nums[i - k]);
            }
            if (i >= k - 1) {
                ans.add(helper.get());
            }
        }

        return ans.stream().mapToLong(Long::longValue).toArray();
    }

    private boolean check(long[] cnt, long val, int r, int k) {
        int n = cnt.length - 1;
        long[] diff = cnt.clone();
        long sum = 0;
        long remaining = k;

        for (int i = 0; i < n; i++) {
            sum += diff[i];
            if (sum < val) {
                long add = val - sum;
                if (remaining < add) {
                    return false;
                }
                remaining -= add;
                int end = Math.min(n, i + 2 * r + 1);
                diff[end] -= add;
                sum += add;
            }
        }
        return true;
    }
    public long maxPower(int[] stations, int r, int k) {
        int n = stations.length;
        long[] cnt = new long[n + 1];

        for (int i = 0; i < n; i++) {
            int left = Math.max(0, i - r);
            int right = Math.min(n, i + r + 1);
            cnt[left] += stations[i];
            cnt[right] -= stations[i];
        }

        long lo = Arrays.stream(stations).min().orElse(0);
        long hi = Arrays.stream(stations).asLongStream().sum() + k;
        long res = 0;

        while (lo <= hi) {
            long mid = lo + (hi - lo) / 2;
            if (check(cnt, mid, r, k)) {
                res = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return res;
    }

    public int minimumOneBitOperations(int n) {
        if (n == 0) {
            return 0;
        }
        int k = 0;
        int curr = 1;
        while (curr*2 <= n) {
            curr *= 2;
            k++;
        }
        return (1 << (k+1))-1-minimumOneBitOperations(n^curr);
    }

    public int minOperations(int[] nums) {
        int res = 0;
        Deque<Integer> deque = new ArrayDeque<>();
        for (int n : nums) {
            while (!deque.isEmpty() && deque.peek()>n) {
                deque.pop();
            }
            if (n == 0) {
                continue;
            }
            while (deque.isEmpty() || deque.peek()<n) {
                deque.push(n);
                res += 1;
            }
        }
        return res;
    }

    public int findMaxForm(String[] strs, int m, int n) {
        Map<Integer, Integer> m0 = new HashMap<>();
        for (int i = 0; i <= strs.length-1; i++) {
            String str = strs[i];
            int cnt = 0;
            for (char c : str.toCharArray()) {
                if (c == '0') {
                    cnt += 1;
                }
            }
            m0.put(i, cnt);
        }
        Map<Integer, Integer> m1 = new HashMap<>();
        for (int i = 0; i <= strs.length-1; i++) {
            String str = strs[i];
            int cnt = 0;
            for (char c : str.toCharArray()) {
                if (c == '1') {
                    cnt += 1;
                }
            }
            m1.put(i, cnt);
        }
        int[][][] dp = new int[strs.length+1][m+1][n+1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                dp[0][i][j] = 0;
            }
        }
        for (int i = 1; i <= strs.length; i++) {
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= n; k++) {
                    int c0 = m0.get(i-1);
                    int c1 = m1.get(i-1);
                    if (c0>j || c1>k) {
                        dp[i][j][k] = dp[i-1][j][k];
                    } else {
                        dp[i][j][k] = Math.max(1+dp[i-1][j-c0][k-c1], dp[i-1][j][k]);
                    }
                }
            }
        }
        return dp[strs.length][m][n];
    }

    public int minOperationsOne(int[] nums) {
        int n = nums.length;
        // Step 1: if global gcd > 1 → impossible
        int g = nums[0];
        for (int i = 1; i < n; i++) {
            g = gcd(g, nums[i]);
        }
        if (g > 1) return -1;
        // Step 2: if we already have 1s
        int countOne = 0;
        for (int num : nums) {
            if (num == 1) countOne++;
        }
        if (countOne > 0) {
            return n - countOne; // each non-1 element takes one operation
        }
        // Step 3: find shortest subarray with gcd == 1
        int minLen = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            int currGcd = nums[i];
            for (int j = i; j < n; j++) {
                currGcd = gcd(currGcd, nums[j]);
                if (currGcd == 1) {
                    minLen = Math.min(minLen, j - i + 1);
                    break; // no need to extend further
                }
            }
        }
        // Step 4: total = (create one 1) + (spread it)
        return (minLen - 1) + (n - 1);
    }

    public int[][] rangeAddQueries(int n, int[][] queries) {
        int[][] res = new int[n][n];
        for (int[] q : queries) {
            int rs = q[0];
            int re = q[2];
            int cs = q[1];
            int ce = q[3];
            for (int i = rs; i <= re; i++) {
                for (int j = cs; j <= ce; j++) {
                    res[i][j] += 1;
                }
            }
        }
        return res;
    }

    public int numberOfSubstrings(String s) {
        int n = s.length();
        int[] pre = new int[n + 1];
        pre[0] = -1;
        for (int i = 0; i < n; i++) {
            if (i == 0 || s.charAt(i - 1) == '0') {
                pre[i + 1] = i;
            } else {
                pre[i + 1] = pre[i];
            }
        }
        int res = 0;
        for (int i = 1; i <= n; i++) {
            int cnt0 = s.charAt(i - 1) == '0' ? 1 : 0;
            int j = i;
            while (j > 0 && cnt0 * cnt0 <= n) {
                int cnt1 = (i - pre[j]) - cnt0;
                if (cnt0 * cnt0 <= cnt1) {
                    res += Math.min(j - pre[j], cnt1 - cnt0 * cnt0 + 1);
                }
                j = pre[j];
                cnt0++;
            }
        }
        return res;
    }

    public boolean kLengthApart(int[] nums, int k) {
        if (nums.length == 1) {
            return true;
        }
        int i = 0;
        while (i<=nums.length-1 && nums[i]!=1) {
            i++;
        }
        if (i == nums.length) {
            return true;
        }
        int dist = 0;
        for (int j = i+1; j <= nums.length-1; j++) {
            if (nums[j] == 1) {
                if (dist < k) {
                    return false;
                }
                dist = 0;
            } else {
                dist += 1;
            }
        }
        return true;
    }

    public boolean isOneBitCharacter(int[] bits) {
        int i = 0;
        while (i < bits.length-1) {
            if (bits[i] == 0) {
                i++;
            } else {
                i += 2;
            }
        }
        return i == bits.length-1;
    }

    public int findFinalValue(int[] nums, int original) {
        Set<Integer> set = new HashSet<>();
        for (int n : nums) {
            set.add(n);
        }
        while (set.contains(original)) {
            original *= 2;
        }
        return original;
    }

    public int minimumOperations(int[] nums) {
        int res = 0;
        for (int n : nums) {
            if (n%3 != 0) {
                res += 1;
            }
        }
        return res;
    }

    public int maxSumDivThree(int[] nums) {
        int total = 0;

        // smallest two numbers with remainder 1 and 2
        int min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        int sec1 = Integer.MAX_VALUE, sec2 = Integer.MAX_VALUE;

        for (int x : nums) {
            total += x;
            int r = x % 3;

            if (r == 1) {
                if (x < min1) {
                    sec1 = min1;
                    min1 = x;
                } else if (x < sec1)
                    sec1 = x;
            } else if (r == 2) {
                if (x < min2) {
                    sec2 = min2;
                    min2 = x;
                } else if (x < sec2)
                    sec2 = x;
            }
        }

        int r = total % 3;
        if (r == 0)
            return total;

        int remove = Integer.MAX_VALUE;

        if (r == 1) {
            // Option 1: remove smallest remainder-1
            if (min1 != Integer.MAX_VALUE)
                remove = min1;
            // Option 2: remove two smallest remainder-2
            if (min2 != Integer.MAX_VALUE && sec2 != Integer.MAX_VALUE)
                remove = Math.min(remove, min2 + sec2);
        } else { // r == 2
            // Option 1: remove smallest remainder-2
            if (min2 != Integer.MAX_VALUE)
                remove = min2;
            // Option 2: remove two smallest remainder-1
            if (min1 != Integer.MAX_VALUE && sec1 != Integer.MAX_VALUE)
                remove = Math.min(remove, min1 + sec1);
        }

        return total - (remove == Integer.MAX_VALUE ? 0 : remove);
    }

    public List<Boolean> prefixesDivBy5(int[] nums) {
        List<Boolean> res = new ArrayList<>();
        int val = 0;
        for (int n : nums) {
            val = ((val<<1)+n)%5;
            if (val == 0) {
                res.add(true);
            } else {
                res.add(false);
            }
        }
        return res;
    }

    public int smallestRepunitDivByK(int k) {
        if (k == 1) {
            return 1;
        }
        int len = 1;
        int rem = 1;
        int op = 10%k;
        Set<Integer> set = new HashSet<>();
        while (!set.contains(rem)) {
            set.add(rem);
            len++;
            rem = (rem*op+1)%k;
            if (rem == 0) {
                return len;
            }
        }
        return -1;
    }

    private static final int MOD = 1000000007;
    public int numberOfPaths(int[][] grid, int k) {
        int m = grid.length;
        int n = grid[0].length;
        long[][][] dp = new long[m + 1][n + 1][k];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == 1 && j == 1) {
                    dp[i][j][grid[0][0] % k] = 1;
                    continue;
                }
                int value = grid[i - 1][j - 1] % k;
                for (int r = 0; r < k; r++) {
                    int prevMod = (r - value + k) % k;
                    dp[i][j][r] =
                            (dp[i - 1][j][prevMod] + dp[i][j - 1][prevMod]) % MOD;
                }
            }
        }
        return (int) dp[m][n][0];
    }

    public long maxSubarraySum(int[] nums, int k) {
        int n = nums.length;
        long prefixSum = 0;
        long maxSum = Long.MIN_VALUE;
        long[] kSum = new long[k];
        Arrays.fill(kSum, Long.MAX_VALUE / 2);
        kSum[k - 1] = 0;
        for (int i = 0; i < n; i++) {
            prefixSum += nums[i];
            maxSum = Math.max(maxSum, prefixSum - kSum[i % k]);
            kSum[i % k] = Math.min(kSum[i % k], prefixSum);
        }
        return maxSum;
    }

    public int minOperations(int[] nums, int k) {
        int sum = 0;
        for (int n : nums) {
            sum += n;
        }
        return sum%k;
    }

    public int minSubarray(int[] nums, int p) {
        long total = 0;
        for (int x : nums)
            total += x;
        int need = (int) (total % p);
        if (need == 0)
            return 0;

        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1); // prefix mod 0 at index -1
        long prefix = 0;
        int ans = nums.length;
        for (int i = 0; i < nums.length; i++) {
            prefix += nums[i];
            int curMod = (int) (prefix % p);
            // We want a previous prefix `prev` such that:
            // prev ≡ (curMod - need + p) % p
            int target = (curMod - need + p) % p;
            if (map.containsKey(target)) {
                ans = Math.min(ans, i - map.get(target));
            }
            map.put(curMod, i);
        }
        return ans == nums.length ? -1 : ans;
    }

    private List<List<Integer>> twoSum(int[] nums, long target, int start) {
        List<List<Integer>> res = new ArrayList<>();
        int lp = start;
        int hp = nums.length-1;
        while (lp < hp) {
            int cs = nums[lp]+nums[hp];
            if (cs<target || (lp>start&&nums[lp]==nums[lp-1])) {
                lp += 1;
            } else if (cs>target || (hp<nums.length-1&&nums[hp]==nums[hp+1])) {
                hp -= 1;
            } else {
                res.add(Arrays.asList(nums[lp++], nums[hp--]));
            }
        }
        return res;
    }
    private List<List<Integer>> kSum(int[] nums, long target, int start, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (start == nums.length) {
            return res;
        }
        if (k == 2) {
            return twoSum(nums, target, start);
        }
        for (int i = start; i <= nums.length-1; i++) {
            if (i==start || nums[i]!=nums[i-1]) {
                for (List<Integer> l : kSum(nums, target-nums[i], i+1, k-1)) {
                    res.add(new ArrayList<>(List.of(nums[i])));
                    res.getLast().addAll(l);
                }
            }
        }
        return res;
    }
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        return kSum(nums, target, 0, 4);
    }

    public int countTrapezoids(int[][] points) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int[] p : points) {
            int height = p[1];
            map.put(height, map.getOrDefault(height, 0)+1);
        }
        if (map.size() < 2) {
            return 0;
        }
        int mod = 1000000007;
        List<Long> list = new ArrayList<>();
        for (int k : map.keySet()) {
            if (map.get(k) >= 2) {
                long np = map.get(k);
                list.add(np*(np-1)/2);
            }
        }
        if (list.size() < 2) {
            return 0;
        }
        long res = list.getFirst()%mod;
        long acc = res%mod;
        for (int i = 1; i <= list.size()-1; i++) {
            long curr = list.get(i)%mod;
            if (i == 1) {
                res *= curr;
            } else {
                res = res + acc*curr;
            }
            res %= mod;
            acc += curr;
        }
        return (int) res%mod;
    }

    public int countCollisions(String directions) {
        if (directions.length() == 1) {
            return 0;
        }
        int res = 0;
        Deque<Character> deque = new ArrayDeque<>();
        for (char c : directions.toCharArray()) {
            if (deque.isEmpty()) {
                deque.push(c);
            } else {
                if (deque.peek() == 'L') {
                    deque.push(c);
                } else if (deque.peek() == 'R') {
                    if (c == 'L') {
                        res += 2;
                        deque.pop();
                        while (!deque.isEmpty()) {
                            char next = deque.peek();
                            if (next == 'R') {
                                res += 1;
                                deque.pop();
                            } else {
                                break;
                            }
                        }
                        deque.push('S');
                    } else if (c == 'R') {
                        deque.push(c);
                    } else {
                        res += 1;
                        deque.pop();
                        while (!deque.isEmpty()) {
                            char next = deque.peek();
                            if (next == 'R') {
                                res += 1;
                                deque.pop();
                            } else {
                                break;
                            }
                        }
                        deque.push('S');
                    }
                } else {
                    if (c == 'L') {
                        res += 1;
                    } else {
                        deque.push(c);
                    }
                }
            }
        }
        return res;
    }
}
