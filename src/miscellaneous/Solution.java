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

            // If memory full, evict the oldest packet
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
        PriorityQueue<Cell> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.height));
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

        // process jobs 1...m-1
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
        // Step 3: find the shortest subarray with gcd == 1
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

    public int countPartitions(int[] nums) {
        int res = 0;
        int ts = 0;
        for (int n : nums) {
            ts += n;
        }
        int ls = 0;
        for (int i = 0; i < nums.length-1; i++) {
            ls += nums[i];
            int rs = ts-ls;
            if (Math.abs(ls-rs)%2 == 0) {
                res += 1;
            }
        }
        return res;
    }

    public int countPartitions(int[] nums, int k) {
        int n = nums.length;
        final int MOD = 1_000_000_007;
        long[] dp = new long[n + 1]; // dp[i] = ways for nums[0..i-1]
        long[] pref = new long[n + 1]; // prefix sum for dp
        dp[0] = 1;
        pref[0] = 1;

        // monotonic deques: store values, not indices
        java.util.Deque<Integer> maxD = new java.util.ArrayDeque<>();
        java.util.Deque<Integer> minD = new java.util.ArrayDeque<>();

        int left = 0;

        for (int right = 0; right < n; right++) {

            // Maintain MAX deque
            while (!maxD.isEmpty() && maxD.peekLast() < nums[right]) {
                maxD.pollLast();
            }
            maxD.addLast(nums[right]);

            // Maintain MIN deque
            while (!minD.isEmpty() && minD.peekLast() > nums[right]) {
                minD.pollLast();
            }
            minD.addLast(nums[right]);

            // Shrink window from the left until valid
            while (!maxD.isEmpty() && !minD.isEmpty() &&
                    maxD.peekFirst() - minD.peekFirst() > k) {

                // If the outgoing element equals the front of max deque, remove it
                if (nums[left] == maxD.peekFirst())
                    maxD.pollFirst();
                assert !minD.isEmpty();
                if (nums[left] == minD.peekFirst())
                    minD.pollFirst();

                left++;
            }

            // Now all segments starting from [left...right] are valid
            // dp[right+1] = sum(dp[left...right])
            long sum = pref[right] - (left == 0 ? 0 : pref[left - 1]);
            sum = (sum % MOD + MOD) % MOD; // normalize

            dp[right + 1] = sum;
            pref[right + 1] = (pref[right] + dp[right + 1]) % MOD;
        }

        return (int) dp[n];
    }

    public int countOdds(int low, int high) {
        if (low%2==0 && high%2==0) {
            return countOdds(low+1, high-1);
        } else if (low%2 == 0) {
            return countOdds(low+1, high);
        } else if (high%2 == 0) {
            return countOdds(low, high-1);
        } else {
            return (high-low)/2+1;
        }
    }

    public int countTriples(int n) {
        if (n == 1) {
            return 0;
        }
        Set<Integer> set = new HashSet<>();
        for (int i = 1; i <= n; i++) {
            set.add(i*i);
        }
        int res = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                int square = i*i+j*j;
                if (set.contains(square)) {
                    res += 1;
                }
            }
        }
        return res;
    }

    public int specialTriplets(int[] nums) {
        final long MOD = 1_000_000_007L;
        int maxVal = 100000;
        int[] freqRight = new int[maxVal + 1];
        for (int x : nums)
            freqRight[x]++;
        int[] freqLeft = new int[maxVal + 1];
        long res = 0;
        for (int mid : nums) {
            // This element is now being processed, remove it from the right side
            freqRight[mid]--;
            int twice = mid * 2;
            if (twice <= maxVal) {
                long left = freqLeft[twice];
                long right = freqRight[twice];
                res = (res + left * right) % MOD;
            }
            // Add current to left side
            freqLeft[mid]++;
        }
        return (int) res;
    }

    public int countPermutations(int[] complexity) {
        for (int i = 1; i <= complexity.length-1; i++) {
            if (complexity[i] <= complexity[0]) {
                return 0;
            }
        }
        int fac = complexity.length-1;
        long res = 1;
        int mod = 1000000007;
        while (fac >= 1) {
            res = ((res%mod)*(fac%mod))%mod;
            fac--;
        }
        return (int) res%mod;
    }

    public int countCoveredBuildings(int n, int[][] buildings) {
        if (n < 3) {
            return 0;
        }
        Map<Integer, Integer> xmax = new HashMap<>();
        Map<Integer, Integer> xmin = new HashMap<>();
        Map<Integer, Integer> ymax = new HashMap<>();
        Map<Integer, Integer> ymin = new HashMap<>();
        for (int[] b : buildings) {
            int x = b[0];
            int y = b[1];
            xmax.put(x, Math.max(xmax.getOrDefault(x, Integer.MIN_VALUE), y));
            xmin.put(x, Math.min(xmin.getOrDefault(x, Integer.MAX_VALUE), y));
            ymax.put(y, Math.max(ymax.getOrDefault(y, Integer.MIN_VALUE), x));
            ymin.put(y, Math.min(ymin.getOrDefault(y, Integer.MAX_VALUE), x));
        }
        int res = 0;
        for (int[] b : buildings) {
            int x = b[0];
            int y = b[1];
            if (x!=ymax.get(y) && x!=ymin.get(y) && y!=xmax.get(x) && y!=xmin.get(x)) {
                res += 1;
            }
        }
        return res;
    }

    public int[] countMentions(int numberOfUsers, List<List<String>> events) {
        events.sort(
                Comparator
                        .comparingInt((List<String> inner) -> Integer.parseInt(inner.get(1)))
                        .thenComparingInt(inner -> inner.getFirst().equals("OFFLINE") ? 0 : 1)
        );
        int[] mention = new int[numberOfUsers];
        int[] ol = new int[numberOfUsers];
        for (List<String> event : events) {
            String status = event.getFirst();
            int time = Integer.parseInt(event.get(1));
            String targets = event.getLast();
            if (status.equals("OFFLINE")) {
                int id = Integer.parseInt(targets);
                ol[id] = time+60;
            } else {
                if (targets.equals("ALL")) {
                    for (int i = 0; i <= mention.length-1; i++) {
                        mention[i] += 1;
                    }
                } else if (targets.equals("HERE")) {
                    for (int i = 0; i <= mention.length-1; i++) {
                        if (time >= ol[i]) {
                            mention[i] += 1;
                        }
                    }
                } else {
                    String[] parts = targets.split(" ");
                    for (String p : parts) {
                        int id = Integer.parseInt(p.substring(2));
                        mention[id] += 1;
                    }
                }
            }
        }
        return mention;
    }

    public List<String> validateCoupons(String[] code, String[] businessLine, boolean[] isActive) {
        List<String> res = new ArrayList<>();
        int n = code.length;
        List<String> ec = new ArrayList<>();
        List<String> gc = new ArrayList<>();
        List<String> pc = new ArrayList<>();
        List<String> rc = new ArrayList<>();
        for (int i = 0; i <= n-1; i++) {
            if (isActive[i] && code[i].matches("[A-Za-z0-9_]+")) {
                switch (businessLine[i]) {
                    case "electronics" -> ec.add(code[i]);
                    case "grocery" -> gc.add(code[i]);
                    case "pharmacy" -> pc.add(code[i]);
                    case "restaurant" -> rc.add(code[i]);
                }
            }
        }
        Collections.sort(ec);
        Collections.sort(gc);
        Collections.sort(pc);
        Collections.sort(rc);
        res.addAll(ec);
        res.addAll(gc);
        res.addAll(pc);
        res.addAll(rc);
        return res;
    }

    public int numberOfWays(String corridor) {
        final int MOD = 1_000_000_007;
        int seatCount = 0;
        for (char c : corridor.toCharArray()) {
            if (c == 'S')
                seatCount++;
        }
        if (seatCount == 0 || seatCount%2!=0)
            return 0;
        long result = 1;
        int seatsSeen = 0;
        int plantsBetween = 0;
        for (char c : corridor.toCharArray()) {
            if (c == 'S') {
                seatsSeen++;
                if (seatsSeen % 2 == 0) {
                    result = (result * (plantsBetween + 1)) % MOD;
                    plantsBetween = 0;
                }
            } else if (seatsSeen!=0 && seatsSeen%2==0) {
                plantsBetween++;
            }
        }
        return (int) result;
    }

    public long getDescentPeriods(int[] prices) {
        if (prices.length == 1) {
            return 1;
        }
        long res = 0;
        int cs = 0;
        int prev = -1;
        for (int i = 0; i <= prices.length-1; i++) {
            if (i == 0) {
                cs = 1;
            } else {
                if (prices[i] == prev-1) {
                    cs += 1;
                } else {
                    cs = 1;
                }
            }
            res += cs;
            prev = prices[i];
        }
        return res;
    }

    static final int NEG = -1_000_000_000;
    List<Integer>[] tree;
    int[] present, future;
    int budget;
    private int[][] dfs(int u) {
        // base dp: before considering children
        int[][] dp = new int[2][budget + 1];
        for (int s = 0; s < 2; s++) {
            Arrays.fill(dp[s], NEG);
            dp[s][0] = 0;
        }
        // merge children
        for (int v : tree[u]) {
            int[][] child = dfs(v);
            int[][] next = new int[2][budget + 1];
            for (int s = 0; s < 2; s++) {
                Arrays.fill(next[s], NEG);
                for (int i = 0; i <= budget; i++) {
                    if (dp[s][i] < 0)
                        continue;
                    for (int j = 0; j + i <= budget; j++) {
                        if (child[s][j] < 0)
                            continue;
                        next[s][i + j] = Math.max(
                                next[s][i + j],
                                dp[s][i] + child[s][j]);
                    }
                }
            }
            dp = next;
        }
        // final dp after deciding whether to buy u
        int[][] res = new int[2][budget + 1];
        for (int s = 0; s < 2; s++)
            Arrays.fill(res[s], NEG);
        // state 0: boss NOT bought
        int cost0 = present[u];
        for (int b = 0; b <= budget; b++) {
            // don't buy u
            res[0][b] = dp[0][b];
            // buy u
            if (b >= cost0) {
                res[0][b] = Math.max(
                        res[0][b],
                        dp[1][b - cost0] + future[u] - cost0);
            }
        }
        // state 1: boss IS bought
        int cost1 = present[u] / 2;
        for (int b = 0; b <= budget; b++) {
            // don't buy u
            res[1][b] = dp[0][b];

            // buy u
            if (b >= cost1) {
                res[1][b] = Math.max(
                        res[1][b],
                        dp[1][b - cost1] + future[u] - cost1);
            }
        }

        return res;
    }
    @SuppressWarnings("unchecked")
    public int maxProfit(int n, int[] present, int[] future, int[][] hierarchy, int budget) {
        this.present = present;
        this.future = future;
        this.budget = budget;
        tree = new ArrayList[n];
        for (int i = 0; i < n; i++)
            tree[i] = new ArrayList<>();
        // build tree (0-based)
        for (int[] e : hierarchy) {
            tree[e[0] - 1].add(e[1] - 1);
        }
        int[][] dpRoot = dfs(0);
        int ans = 0;
        for (int b = 0; b <= budget; b++) {
            ans = Math.max(ans, dpRoot[0][b]);
        }
        return ans;
    }

    public long maximumProfit(int[] prices, int k) {
        final long NEG_INF = Long.MIN_VALUE / 4;
        long[] flat = new long[k + 1];
        long[] lon = new long[k + 1];
        long[] sh = new long[k + 1];
        Arrays.fill(flat, NEG_INF);
        Arrays.fill(lon, NEG_INF);
        Arrays.fill(sh, NEG_INF);
        flat[0] = 0;

        for (int price : prices) {
            long[] nFlat = new long[k + 1];
            long[] nLon = new long[k + 1];
            long[] nSh = new long[k + 1];
            Arrays.fill(nFlat, NEG_INF);
            Arrays.fill(nLon, NEG_INF);
            Arrays.fill(nSh, NEG_INF);
            for (int t = 0; t <= k; t++) {
                // Stay flat
                nFlat[t] = Math.max(nFlat[t], flat[t]);
                // Close positions
                if (t > 0) {
                    if (lon[t - 1] != NEG_INF) {
                        nFlat[t] = Math.max(nFlat[t], lon[t - 1] + price);
                    }
                    if (sh[t - 1] != NEG_INF) {
                        nFlat[t] = Math.max(nFlat[t], sh[t - 1] - price);
                    }
                }
                // Keep or open long
                if (flat[t] != NEG_INF) {
                    nLon[t] = Math.max(nLon[t], flat[t] - price);
                }
                nLon[t] = Math.max(nLon[t], lon[t]);
                // Keep or open short
                if (flat[t] != NEG_INF) {
                    nSh[t] = Math.max(nSh[t], flat[t] + price);
                }
                nSh[t] = Math.max(nSh[t], sh[t]);
            }
            flat = nFlat;
            lon = nLon;
            sh = nSh;
        }
        long ans = 0;
        for (int t = 0; t <= k; t++) {
            ans = Math.max(ans, flat[t]);
        }
        return ans;
    }

    public long maxProfit(int[] prices, int[] strategy, int k) {
        long sum = 0;
        for (int i = 0; i <= prices.length-1; i++) {
            sum += (long) prices[i] * strategy[i];
        }
        long res = sum;
        long prev = 0;
        for (int i = 0; i <= prices.length-1; i++) {
            if (i >= k/2) {
                if (i <= k-1) {
                    prev += prices[i];
                } else {
                    prev += (long) prices[i] * strategy[i];
                }
            }
        }
        res = Math.max(res, prev);
        for (int i = 1; i <= prices.length-k; i++) {
            long curr = prev;
            curr += (long) prices[i - 1] * strategy[i-1];
            int ci = k/2-1+i;
            curr -= prices[ci];
            int ti = i+k-1;
            curr -= (long) prices[ti] * strategy[ti];
            curr += prices[ti];
            res = Math.max(res, curr);
            prev = curr;
        }
        return res;
    }

    static class UnionFind {
        int[] parent;

        UnionFind(int n) {
            parent = new int[n];
            for (int i = 0; i < n; i++) parent[i] = i;
        }

        int find(int node) {
            if (parent[node] != node)
                parent[node] = find(parent[node]);
            return parent[node];
        }

        void union(int x, int y) {
            parent[find(x)] = find(y);
        }

        void reset(int x) {
            parent[x] = x;
        }
    }
    public List<Integer> findAllPeople(int n, int[][] meetings, int firstPerson) {
        Arrays.sort(meetings, Comparator.comparingInt(a -> a[2]));
        boolean[] knows = new boolean[n];
        knows[0] = true;
        knows[firstPerson] = true;
        UnionFind uf = new UnionFind(n);
        int i = 0;
        while (i < meetings.length) {
            int time = meetings[i][2];
            List<int[]> batch = new ArrayList<>();
            while (i < meetings.length && meetings[i][2] == time) {
                batch.add(meetings[i]);
                uf.union(meetings[i][0], meetings[i][1]);
                i++;
            }
            // Track which roots should spread the secret
            Set<Integer> secretRoots = new HashSet<>();
            for (int[] m : batch) {
                if (knows[m[0]] || knows[m[1]]) {
                    secretRoots.add(uf.find(m[0]));
                    secretRoots.add(uf.find(m[1]));
                }
            }
            // Spread secret to entire components
            for (int[] m : batch) {
                if (secretRoots.contains(uf.find(m[0]))) {
                    knows[m[0]] = true;
                    knows[m[1]] = true;
                }
            }
            // Reset DSU for next time
            for (int[] m : batch) {
                uf.reset(m[0]);
                uf.reset(m[1]);
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int p = 0; p < n; p++) {
            if (knows[p]) res.add(p);
        }
        return res;
    }

    public int minDeletionSize(String[] strs) {
        int res = 0;
        int cn = strs[0].length();
        for (int i = 0; i <= cn-1; i++) {
            char cc = strs[0].charAt(i);
            for (int j = 1; j <= strs.length-1; j++) {
                char nc = strs[j].charAt(i);
                if (nc-cc < 0) {
                    res += 1;
                    break;
                } else {
                    cc = nc;
                }
            }
        }
        return res;
    }

    public int minDeletionSizeII(String[] strs) {
        int n = strs.length;
        int m = strs[0].length();
        int deleteCount = 0;
        boolean[] sorted = new boolean[n - 1];
        for (int col = 0; col < m; col++) {
            boolean needDelete = false;
            for (int row = 0; row < n - 1; row++) {
                if (!sorted[row] && strs[row].charAt(col) > strs[row + 1].charAt(col)) {
                    needDelete = true;
                    break;
                }
            }
            if (needDelete) {
                deleteCount++;
            } else {
                for (int row = 0; row < n - 1; row++) {
                    if (!sorted[row] && strs[row].charAt(col) < strs[row + 1].charAt(col)) {
                        sorted[row] = true;
                    }
                }
            }
        }
        return deleteCount;
    }

    private int lastIndexLessThan(int[] arr, int x) {
        int lo = 0, hi = arr.length - 1;
        int res = -1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (arr[mid] < x) {
                res = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return res;
    }
    public int maxTwoEvents(int[][] events) {
        // Sort events by end time
        Arrays.sort(events, Comparator.comparingInt(a -> a[1]));
        int n = events.length;
        int[] ends = new int[n];
        int[] best = new int[n];
        for (int i = 0; i < n; i++) {
            ends[i] = events[i][1];
            best[i] = events[i][2];
            if (i > 0) {
                best[i] = Math.max(best[i], best[i - 1]);
            }
        }
        int ans = 0;
        for (int[] event : events) {
            int start = event[0];
            int value = event[2];
            // Find the last event that ends before this one starts
            int idx = lastIndexLessThan(ends, start);
            int candidate = value + (idx >= 0 ? best[idx] : 0);
            ans = Math.max(ans, candidate);
        }
        return ans;
    }

    public int minimumBoxes(int[] apple, int[] capacity) {
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b)->b-a);
        for (int c : capacity) {
            pq.offer(c);
        }
        int res = 0;
        int sum = 0;
        for (int a : apple) {
            sum += a;
        }
        while (sum > 0) {
            assert  pq.peek()!=null;
            int curr = pq.poll();
            res += 1;
            sum -= curr;
        }
        return res;
    }

    public long maximumHappinessSum(int[] happiness, int k) {
        List<Integer> list = new ArrayList<>();
        for (int h : happiness) {
            list.add(h);
        }
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b)->b-a);
        pq.addAll(list);
        int cnt = 0;
        long res = 0;
        while (cnt < k) {
            assert pq.peek() != null;
            int ch = pq.poll();
            ch -= cnt;
            ch = Math.max(ch, 0);
            res += ch;
            cnt += 1;
        }
        return res;
    }

    public int bestClosingTime(String customers) {
        int n = customers.length()+1;
        int[] pre = new int[n];
        for (int i = 1; i <= n-1; i++) {
            if (customers.charAt(i-1) == 'N') {
                pre[i] = pre[i-1]+1;
            } else {
                pre[i] = pre[i-1];
            }
        }
        int[] post = new int[n];
        for (int i = n-2; i >= 0; i--) {
            if (customers.charAt(i) == 'Y') {
                post[i] = post[i+1]+1;
            } else {
                post[i] = post[i+1];
            }
        }
        int bp = Integer.MAX_VALUE;
        int bh = 0;
        int ch = 0;
        while (ch <= customers.length()) {
            int cp = pre[ch]+post[ch];
            if (cp < bp) {
                bp = cp;
                bh = ch;
            }
            ch++;
        }
        return bh;
    }

    public int mostBooked(int n, int[][] meetings) {
        Map<Integer, Long> availTime = new HashMap<>();
        Map<Integer, Integer> meetCnt = new HashMap<>();
        for (int i = 0; i < n; i++) {
            availTime.put(i, (long) 0);
            meetCnt.put(i, 0);
        }
        Arrays.sort(meetings, Comparator.comparingInt(a -> a[0]));
        for (int[] m : meetings) {
            int start = m[0];
            int dur = m[1]-m[0];
            boolean hasAvail = false;
            for (int i = 0; i < n; i++) {
                if (availTime.get(i) <= start) {
                    availTime.put(i, (long) m[1]);
                    meetCnt.put(i, meetCnt.get(i)+1);
                    hasAvail = true;
                    break;
                }
            }
            if (!hasAvail) {
                long earlyEnd = Long.MAX_VALUE;
                int earlyId = 0;
                for (int i = 0; i < n; i++) {
                    if (availTime.get(i) < earlyEnd) {
                        earlyEnd = availTime.get(i);
                        earlyId = i;
                    }
                }
                availTime.put(earlyId, earlyEnd+dur);
                meetCnt.put(earlyId, meetCnt.get(earlyId)+1);
            }
        }
        int mostCnt = Integer.MIN_VALUE;
        int mostId = 0;
        for (int i = 0; i < n; i++) {
            if (meetCnt.get(i) > mostCnt) {
                mostCnt = meetCnt.get(i);
                mostId = i;
            }
        }
        return mostId;
    }

    public int countNegatives(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        if (m > n) {
            for (int i = 0; i < n; i++) {
                int idx = -1;
                int up = 0;
                int down = m-1;
                while (up <= down) {
                    int mid = up+(down-up)/2;
                    if (grid[mid][i] < 0) {
                        down = mid-1;
                        idx = mid;
                    } else {
                        up = mid+1;
                    }
                }
                if (idx != -1) {
                    res = res+m-idx;
                }
            }
        } else {
            for (int[] ints : grid) {
                int idx = -1;
                int left = 0;
                int right = n - 1;
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (ints[mid] < 0) {
                        right = mid - 1;
                        idx = mid;
                    } else {
                        left = mid + 1;
                    }
                }
                if (idx != -1) {
                    res = res + n - idx;
                }
            }
        }
        return res;
    }

    private final int[][] transitions = new int[6][6];
    private final Map<String, Boolean> memo = new HashMap<>();
    private boolean buildNextRow(char[] cur, int idx, char[] next) {
        if (idx == next.length) {
            return canBuild(new String(next));
        }
        int a = cur[idx] - 'A';
        int b = cur[idx + 1] - 'A';
        int mask = transitions[a][b];
        if (mask == 0)
            return false;
        while (mask != 0) {
            int bit = mask & -mask; // lowest set bit
            int c = Integer.numberOfTrailingZeros(bit);
            next[idx] = (char) ('A' + c);
            if (buildNextRow(cur, idx + 1, next))
                return true;
            mask -= bit;
        }
        return false;
    }
    private boolean canBuild(String bottom) {
        if (bottom.length() == 1)
            return true;
        Boolean cached = memo.get(bottom);
        if (cached != null)
            return cached;
        char[] cur = bottom.toCharArray();
        char[] next = new char[cur.length - 1];
        boolean ans = buildNextRow(cur, 0, next);
        memo.put(bottom, ans);
        return ans;
    }
    public boolean pyramidTransition(String bottom, List<String> allowed) {
        for (String s : allowed) {
            int a = s.charAt(0) - 'A';
            int b = s.charAt(1) - 'A';
            int c = s.charAt(2) - 'A';
            transitions[a][b] |= (1 << c);
        }
        return canBuild(bottom);
    }

    public int numMagicSquaresInside(int[][] grid) {
        if (grid.length<3 || grid[0].length<3) {
            return 0;
        }
        int res = 0;
        for (int i = 0; i < grid.length-2; i++) {
            for (int j = 0; j < grid[0].length-2; j++) {
                Set<Integer> set = new HashSet<>();
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        int cn = grid[i+r][j+c];
                        if (cn>=1 && cn<=9) {
                            set.add(cn);
                        }
                    }
                }
                if (set.size() == 9) {
                    int rs1 = grid[i][j]+grid[i][j+1]+grid[i][j+2];
                    int rs2 = grid[i+1][j]+grid[i+1][j+1]+grid[i+1][j+2];
                    int rs3 = grid[i+2][j]+grid[i+2][j+1]+grid[i+2][j+2];
                    int cs1 = grid[i][j]+grid[i+1][j]+grid[i+2][j];
                    int cs2 = grid[i][j+1]+grid[i+1][j+1]+grid[i+2][j+1];
                    int cs3 = grid[i][j+2]+grid[i+1][j+2]+grid[i+2][j+2];
                    int ds1 = grid[i][j]+grid[i+1][j+1]+grid[i+2][j+2];
                    int ds2 = grid[i][j+2]+grid[i+1][j+1]+grid[i+2][j];
                    if (rs1==rs2 && rs2==rs3 &&
                            rs3==cs1 && cs1==cs2 && cs2==cs3 &&
                            cs3==ds1 && ds1==ds2) {
                        res += 1;
                    }
                }
            }
        }
        return res;
    }

    public int[] plusOne(int[] digits) {
        for (int i = digits.length-1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i] += 1;
                return digits;
            } else {
                digits[i] = 0;
            }
        }
        int[] res = new int[digits.length+1];
        res[0] = 1;
        return res;
    }

    public int repeatedNTimes(int[] nums) {
        for (int a = 1; a <= 3; a++) {
            for (int i = 0; i <= nums.length-1-a; i++) {
                if (nums[i] == nums[i+a]) {
                    return nums[i];
                }
            }
        }
        return -1;
    }

    public int numOfWays(int n) {
        int mod = 1_000_000_007;
        long va = 6;
        long vb = 6;
        for (int i = 2; i <= n; i++) {
            long na = (va*2+vb*2)%mod;
            long nb = (va*2+vb*3)%mod;
            va = na;
            vb = nb;
        }
        return (int) (va+vb)%mod;
    }

    public int sumFourDivisors(int[] nums) {
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int n : nums) {
            if (map.containsKey(n)) {
                res = res + map.get(n);
            } else {
                if (n <= 5) {
                    map.put(n, 0);
                    continue;
                }
                int fd = -1;
                for (int i = 2; i <= (int) Math.sqrt(n); i++) {
                    if (n % i == 0) {
                        if (fd != -1) {
                            fd = -1;
                            break;
                        } else {
                            fd = i;
                        }
                    }
                }
                if (fd != -1 && fd != n/fd) {
                    int sum = 1 + n + fd + n/fd;
                    res = res + sum;
                    map.put(n, sum);
                } else {
                    map.put(n, 0);
                }
            }
        }
        return res;
    }

    public long maxMatrixSum(int[][] matrix) {
        int np = 0;
        for (int[] row : matrix) {
            for (int r : row) {
                if (r <= 0) {
                    np++;
                }
            }
        }
        long res = 0;
        if (np % 2 == 0) {
            for (int[] row : matrix) {
                for (int r : row) {
                    res += Math.abs(r);
                }
            }
        } else {
            int min = Integer.MAX_VALUE;
            int mr = 0;
            int mc = 0;
            for (int i = 0; i <= matrix.length-1; i++) {
                for (int j = 0; j <= matrix[0].length-1; j++) {
                    if (Math.abs(matrix[i][j]) < min) {
                        min = Math.abs(matrix[i][j]);
                        mr = i;
                        mc = j;
                    }
                }
            }
            for (int i = 0; i <= matrix.length-1; i++) {
                for (int j = 0; j <= matrix[0].length-1; j++) {
                    if (i==mr && j==mc) {
                        res -= Math.abs(matrix[i][j]);
                    } else {
                        res += Math.abs(matrix[i][j]);
                    }
                }
            }
        }
        return res;
    }

    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int val) {
            this.val = val;
        }
    }
    public int maxLevelSum(TreeNode root) {
        int ms = Integer.MIN_VALUE;
        int ml = 0;
        int cl = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            cl += 1;
            int cs = 0;
            int cn = queue.size();
            while (cn > 0) {
                TreeNode node = queue.poll();
                assert node != null;
                cs += node.val;
                cn--;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            if (cs > ms) {
                ms = cs;
                ml = cl;
            }
        }
        return ml;
    }
    public int maxProduct(TreeNode root) {
        // 1) Get total sum (iterative DFS)
        long total = 0;
        Deque<TreeNode> st = new ArrayDeque<>();
        st.push(root);
        while (!st.isEmpty()) {
            TreeNode node = st.pop();
            total += node.val;
            if (node.left != null)
                st.push(node.left);
            if (node.right != null)
                st.push(node.right);
        }
        // 2) Postorder to compute subtree sums once, track max product.
        Map<TreeNode, Long> subSum = new IdentityHashMap<>(); // fast, node identity keys
        long best = 0;
        Deque<Object[]> stack = new ArrayDeque<>();
        stack.push(new Object[] { root, false });
        while (!stack.isEmpty()) {
            Object[] top = stack.pop();
            TreeNode node = (TreeNode) top[0];
            boolean visited = (boolean) top[1];
            if (!visited) {
                stack.push(new Object[] { node, true });
                if (node.left != null)
                    stack.push(new Object[] { node.left, false });
                if (node.right != null)
                    stack.push(new Object[] { node.right, false });
            } else {
                long left = node.left == null ? 0 : subSum.get(node.left);
                long right = node.right == null ? 0 : subSum.get(node.right);
                long s = left + right + node.val;
                subSum.put(node, s);
                if (node != root) {
                    long product = s * (total - s);
                    if (product > best)
                        best = product;
                }
            }
        }
        long MOD = 1000000007;
        return (int) (best % MOD);
    }
    private boolean isUpper(TreeNode node, List<TreeNode> target) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(node);
        Set<Integer> expl = new HashSet<>();
        while (!queue.isEmpty()) {
            TreeNode curr = queue.poll();
            expl.add(curr.val);
            if (curr.left != null) {
                queue.offer(curr.left);
            }
            if (curr.right != null) {
                queue.offer(curr.right);
            }
        }
        for (TreeNode t : target) {
            if (!expl.contains(t.val)) {
                return false;
            }
        }
        return true;
    }
    public TreeNode subtreeWithAllDeepest(TreeNode root) {
        if (root.left==null && root.right==null) {
            return root;
        }
        Map<Integer, List<TreeNode>> levels = new HashMap<>();
        int lvl = 1;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int cnt = queue.size();
            List<TreeNode> list = new ArrayList<>(cnt);
            while (cnt > 0) {
                TreeNode curr = queue.poll();
                list.add(curr);
                assert curr != null;
                if (curr.left != null) {
                    queue.offer(curr.left);
                }
                if (curr.right != null) {
                    queue.offer(curr.right);
                }
                cnt--;
            }
            levels.put(lvl, list);
            lvl++;
        }
        List<TreeNode> deep = levels.get(lvl-1);
        if (deep.size() == 1) {
            return deep.getFirst();
        } else {
            for (int l = lvl-2; l >= 2; l--) {
                List<TreeNode> curr = levels.get(l);
                for (TreeNode node : curr) {
                    if (isUpper(node, deep)) {
                        return node;
                    }
                }
            }
            return root;
        }
    }
    private int height(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int left = height(node.left);
        if (left == -1) {
            return -1;
        }
        int right = height(node.right);
        if (right == -1) {
            return -1;
        }
        if (Math.abs(left-right) > 1) {
            return -1;
        }
        return Math.max(left, right)+1;
    }
    public boolean isBalanced(TreeNode root) {
        int bal = height(root);
        return bal != -1;
    }
    private void inorder(TreeNode node, List<TreeNode> list) {
        if (node != null) {
            inorder(node.left, list);
            list.add(node);
            inorder(node.right, list);
        }
    }
    private TreeNode build(List<TreeNode> list, int lp, int rp) {
        int mp = (lp+rp)/2;
        TreeNode root = list.get(mp);
        root.left = null;
        root.right = null;
        if (mp-1 >= lp) {
            root.left = build(list, lp, mp-1);
        }
        if (mp+1 <= rp) {
            root.right = build(list, mp+1, rp);
        }
        return root;
    }
    public TreeNode balanceBST(TreeNode root) {
        List<TreeNode> list = new ArrayList<>();
        inorder(root, list);
        return build(list, 0, list.size()-1);
    }

    public int maxDotProduct(int[] nums1, int[] nums2) {
        int l1 = nums1.length, l2 = nums2.length;
        long NEG = Long.MIN_VALUE / 4;
        long[][] dp = new long[l1 + 1][l2 + 1];
        for (int i = 0; i <= l1; i++)
            dp[i][0] = NEG;
        for (int j = 0; j <= l2; j++)
            dp[0][j] = NEG;
        for (int i = 1; i <= l1; i++) {
            for (int j = 1; j <= l2; j++) {
                long prod = (long) nums1[i - 1] * nums2[j - 1];
                long addon = (dp[i - 1][j - 1] == NEG) ? NEG : dp[i - 1][j - 1] + prod;
                dp[i][j] = Math.max(Math.max(dp[i - 1][j], dp[i][j - 1]),
                        Math.max(prod, addon));
            }
        }
        return (int) dp[l1][l2];
    }

    public int minimumDeleteSum(String s1, String s2) {
        int r = s1.length();
        int c = s2.length();
        int[][] dp = new int[r+1][c+1];
        for (int j = 0; j <= c; j++) {
            dp[0][j] = 0;
        }
        for (int i = 0; i <= r; i++) {
            dp[i][0] = 0;
        }
        for (int i = 1; i <= r; i++) {
            for (int j = 1; j <= c; j++) {
                if (s1.charAt(i-1) == s2.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1]+(int) s1.charAt(i-1);
                } else {
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
                }
            }
        }
        int total = 0;
        for (char ch : s1.toCharArray()) {
            total += ch;
        }
        for (char ch : s2.toCharArray()) {
            total += ch;
        }
        return total-2*dp[r][c];
    }

    private int largestRectangleArea(int[] heights) {
        int n = heights.length;
        Deque<Integer> st = new ArrayDeque<>();
        int best = 0;
        for (int i = 0; i <= n; i++) {
            int cur = (i == n) ? 0 : heights[i];
            while (!st.isEmpty() && heights[st.peek()] > cur) {
                int h = heights[st.pop()];
                int leftLessIdx = st.isEmpty() ? -1 : st.peek();
                int width = i - leftLessIdx - 1;
                best = Math.max(best, h * width);
            }
            st.push(i);
        }

        return best;
    }
    public int maximalRectangle(char[][] matrix) {
        int cols = matrix[0].length;
        int[] heights = new int[cols];
        int best = 0;
        for (char[] chars : matrix) {
            for (int c = 0; c < cols; c++) {
                if (chars[c] == '1') heights[c] += 1;
                else heights[c] = 0;
            }
            best = Math.max(best, largestRectangleArea(heights));
        }
        return best;
    }

    public int minTimeToVisitAllPoints(int[][] points) {
        int n = points.length;
        if (n == 1) {
            return 0;
        }
        int res = 0;
        int cx = points[0][0];
        int cy = points[0][1];
        for (int i = 1; i <= n-1; i++) {
            int nx = points[i][0];
            int ny = points[i][1];
            if (nx==cx || ny==cy) {
                res += (Math.abs(nx-cx)+Math.abs(ny-cy));
            } else {
                int dx = Math.abs(nx-cx);
                int dy = Math.abs(ny-cy);
                int md = Math.min(dx, dy);
                res += (dx+dy-md);
            }
            cx = nx;
            cy = ny;
        }
        return res;
    }

    private double areaBelow(int[][] squares, double Y) {
        double sum = 0.0;
        for (int[] s : squares) {
            double y = s[1];
            double l = s[2];
            if (Y <= y) {
                sum += 0.0;
            } else if (Y >= y + l) {
                sum += l * l;
            } else {
                sum += l * (Y - y);
            }
        }
        return sum;
    }
    public double separateSquares(int[][] squares) {
        double totalArea = 0.0;
        double low = Double.POSITIVE_INFINITY;
        double high = Double.NEGATIVE_INFINITY;
        for (int[] s : squares) {
            double y = s[1];
            double l = s[2];
            totalArea += l * l;
            low = Math.min(low, y);
            high = Math.max(high, y + l);
        }
        double half = totalArea / 2.0;
        // Binary search for Y such that area below Y == half
        // 60-80 iterations is plenty for 1e-5 accuracy in double
        for (int it = 0; it < 80; it++) {
            double mid = low + (high - low) / 2.0;
            double below = areaBelow(squares, mid);
            if (below < half) {
                low = mid;
            } else {
                high = mid;
            }
        }
        // low and high are extremely close; either is fine
        return high;
    }

    static class Event {
        long y;
        long x1, x2;
        int type; // +1 add, -1 remove

        Event(long y, long x1, long x2, int type) {
            this.y = y;
            this.x1 = x1;
            this.x2 = x2;
            this.type = type;
        }
    }
    static class Slab {
        long y0, y1;
        long coveredLen; // union length on x in this slab
        double cumStart; // cumulative area before this slab
        double area; // area contributed by this slab

        Slab(long y0, long y1, long coveredLen, double cumStart, double area) {
            this.y0 = y0;
            this.y1 = y1;
            this.coveredLen = coveredLen;
            this.cumStart = cumStart;
            this.area = area;
        }
    }
    static class SegTree {
        final long[] xs; // unique sorted coordinates
        final int nSeg; // number of elementary segments = xs.length - 1
        final int[] cover; // cover count
        final long[] len; // covered length in this node

        SegTree(long[] xs) {
            this.xs = xs;
            this.nSeg = xs.length - 1;
            int size = 4 * Math.max(1, nSeg);
            this.cover = new int[size];
            this.len = new long[size];
        }

        void update(int ql, int qr, int delta) {
            if (ql > qr)
                return;
            update(1, 0, nSeg - 1, ql, qr, delta);
        }

        private void update(int idx, int l, int r, int ql, int qr, int delta) {
            if (ql <= l && r <= qr) {
                cover[idx] += delta;
                pull(idx, l, r);
                return;
            }
            int mid = (l + r) >>> 1;
            if (ql <= mid)
                update(idx << 1, l, mid, ql, qr, delta);
            if (qr > mid)
                update(idx << 1 | 1, mid + 1, r, ql, qr, delta);
            pull(idx, l, r);
        }

        private void pull(int idx, int l, int r) {
            if (cover[idx] > 0) {
                // fully covered
                len[idx] = xs[r + 1] - xs[l];
            } else if (l == r) {
                len[idx] = 0;
            } else {
                len[idx] = len[idx << 1] + len[idx << 1 | 1];
            }
        }

        long coveredLen() {
            return len[1];
        }
    }
    private static long[] unique(long[] arr) {
        int m = arr.length;
        long[] tmp = new long[m];
        int k = 0;
        for (int i = 0; i < m; i++) {
            if (i == 0 || arr[i] != arr[i - 1])
                tmp[k++] = arr[i];
        }
        return Arrays.copyOf(tmp, k);
    }
    public double separateSquaresII(int[][] squares) {
        int n = squares.length;
        // Build events and collect x coords for compression.
        Event[] events = new Event[2 * n];
        long[] xCoords = new long[2 * n];
        int ei = 0, xi = 0;

        for (int[] s : squares) {
            long x = s[0], y = s[1], l = s[2];
            long x2 = x + l;
            long y2 = y + l;
            events[ei++] = new Event(y, x, x2, +1);
            events[ei++] = new Event(y2, x, x2, -1);
            xCoords[xi++] = x;
            xCoords[xi++] = x2;
        }

        Arrays.sort(events, Comparator.comparingLong(a -> a.y));

        Arrays.sort(xCoords);
        long[] xs = unique(xCoords);
        if (xs.length <= 1) {
            // All squares would have zero width (not possible with l>=1), but guard anyway.
            return events[0].y;
        }

        SegTree st = new SegTree(xs);

        // Helper: map x interval to indices in xs.
        // Leaf i represents [xs[i], xs[i+1]] for i in [0..xs.length-2]
        // For interval [x1, x2), we update i in [idx(x1) .. idx(x2)-1]
        // since x2 is an endpoint coordinate.
        Map<Long, Integer> xIndex = new HashMap<>(xs.length * 2);
        for (int i = 0; i < xs.length; i++)
            xIndex.put(xs[i], i);

        List<Slab> slabs = new ArrayList<>();
        long prevY = events[0].y;
        double cum = 0.0;

        int i = 0;
        while (i < events.length) {
            long y = events[i].y;

            long deltaY = y - prevY;
            if (deltaY > 0) {
                long covered = st.coveredLen();
                if (covered > 0) {
                    double area = (double) covered * (double) deltaY;
                    slabs.add(new Slab(prevY, y, covered, cum, area));
                    cum += area;
                }
                prevY = y;
            }

            // Apply all events at this y
            while (i < events.length && events[i].y == y) {
                Event e = events[i];
                int l = xIndex.get(e.x1);
                int r = xIndex.get(e.x2);
                // update segments [l .. r-1]
                st.update(l, r - 1, e.type);
                i++;
            }
        }

        double total = cum;
        double half = total / 2.0;

        // Find Y in slabs
        for (Slab s : slabs) {
            double end = s.cumStart + s.area;
            if (half <= end + 1e-12) { // tiny tolerance for floating error
                if (half <= s.cumStart + 1e-12)
                    return (double) s.y0;
                // Interpolate inside slab
                double need = half - s.cumStart;
                return (double) s.y0 + need / (double) s.coveredLen;
            }
        }

        // If half==total (can happen with precision), return top-most.
        return (double) prevY;
    }

    public int maximizeSquareHoleArea(int n, int m, int[] hBars, int[] vBars) {
        Arrays.sort(hBars);
        int ci = hBars[0];
        int chs = 1;
        int bhs = 1;
        for (int i = 1; i <= hBars.length-1; i++) {
            int ni = hBars[i];
            if (ni == ci+1) {
                chs += 1;
            } else {
                bhs = Math.max(bhs, chs);
                chs = 1;
            }
            ci = ni;
        }
        bhs = Math.max(bhs, chs);
        Arrays.sort(vBars);
        ci = vBars[0];
        int cvs = 1;
        int bvs = 1;
        for (int i = 1; i <= vBars.length-1; i++) {
            int ni = vBars[i];
            if (ni == ci+1) {
                cvs += 1;
            } else {
                bvs = Math.max(bvs, cvs);
                cvs = 1;
            }
            ci = ni;
        }
        bvs = Math.max(bvs, cvs);
        int side = Math.min(bhs, bvs);
        return (side+1)*(side+1);
    }

    private static Set<Integer> getHset(int m, int[] hFences) {
        Set<Integer> hset = new HashSet<>();
        for (int i = 0; i <= hFences.length; i++) {
            for (int j = i+1; j <= hFences.length+1; j++) {
                if (i == 0) {
                    if (j != hFences.length+1) {
                        hset.add(hFences[j-1]-1);
                    } else {
                        hset.add(m -1);
                    }
                } else {
                    if (j != hFences.length+1) {
                        hset.add(hFences[j-1]- hFences[i-1]);
                    } else {
                        hset.add(m - hFences[i-1]);
                    }
                }
            }
        }
        return hset;
    }
    public int maximizeSquareArea(int m, int n, int[] hFences, int[] vFences) {
        Arrays.sort(hFences);
        Set<Integer> hset = getHset(m, hFences);
        int bd = 0;
        Arrays.sort(vFences);
        for (int i = 0; i <= vFences.length; i++) {
            for (int j = i+1; j <= vFences.length+1; j++) {
                int diff;
                if (i == 0) {
                    if (j != vFences.length+1) {
                        diff = vFences[j-1]-1;
                    } else {
                        diff = n-1;
                    }
                } else {
                    if (j != vFences.length+1) {
                        diff = vFences[j-1]-vFences[i-1];
                    } else {
                        diff = n-vFences[i-1];
                    }
                }
                if (hset.contains(diff)) {
                    bd = Math.max(bd, diff);
                }
            }
        }
        if (bd == 0) {
            return -1;
        } else {
            long ans = (long) bd*bd%1000000007;
            return (int) ans;
        }
    }

    public long largestSquareArea(int[][] bottomLeft, int[][] topRight) {
        long bl = 0;
        int n = bottomLeft.length;
        for (int i = 0; i <= n-2; i++) {
            int xib = bottomLeft[i][0];
            int yib = bottomLeft[i][1];
            int xit = topRight[i][0];
            int yit = topRight[i][1];
            for (int j = i+1; j <= n-1; j++) {
                int xjb = bottomLeft[j][0];
                int yjb = bottomLeft[j][1];
                int xjt = topRight[j][0];
                int yjt = topRight[j][1];
                int w = Math.min(xit, xjt)-Math.max(xib, xjb);
                int h = Math.min(yit, yjt)-Math.max(yib, yjb);
                if (w>0 && h>0) {
                    bl = Math.max(bl, Math.min(w, h));
                }
            }
        }
        return bl*bl;
    }

    public int largestMagicSquare(int[][] grid) {
        int m = grid.length, n = grid[0].length;

        // Prefix sums for rows and columns:
        // rowPref[r][c+1] = sum(grid[r][0..c])
        // colPref[r+1][c] = sum(grid[0..r][c])
        long[][] rowPref = new long[m][n + 1];
        long[][] colPref = new long[m + 1][n];

        for (int r = 0; r < m; r++) {
            long run = 0;
            for (int c = 0; c < n; c++) {
                run += grid[r][c];
                rowPref[r][c + 1] = run;
            }
        }

        for (int c = 0; c < n; c++) {
            long run = 0;
            for (int r = 0; r < m; r++) {
                run += grid[r][c];
                colPref[r + 1][c] = run;
            }
        }

        // Diagonal prefix sums:
        // diag1Pref[r+1][c+1] accumulates along top-left -> bottom-right:
        // diag1Pref[r+1][c+1] = grid[r][c] + diag1Pref[r][c]
        long[][] diag1Pref = new long[m + 1][n + 1];

        // diag2Pref[r+1][c] accumulates along top-right -> bottom-left:
        // diag2Pref[r+1][c] = grid[r][c] + diag2Pref[r][c+1]
        long[][] diag2Pref = new long[m + 1][n + 1];

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                diag1Pref[r + 1][c + 1] = diag1Pref[r][c] + grid[r][c];
            }
            for (int c = n - 1; c >= 0; c--) {
                diag2Pref[r + 1][c] = diag2Pref[r][c + 1] + grid[r][c];
            }
        }

        // Helper lambdas (implemented as local methods via private functions not allowed here).
        // We'll inline computations for performance/clarity.

        int maxK = Math.min(m, n);

        // Try larger k first, return immediately when found.
        for (int k = maxK; k >= 2; k--) {
            for (int r0 = 0; r0 + k <= m; r0++) {
                int r1 = r0 + k - 1;
                for (int c0 = 0; c0 + k <= n; c0++) {
                    int c1 = c0 + k - 1;

                    // Target sum = first row sum of the kxk square
                    long target = rowPref[r0][c1 + 1] - rowPref[r0][c0];

                    // Check both diagonals
                    long d1 = diag1Pref[r1 + 1][c1 + 1] - diag1Pref[r0][c0];
                    if (d1 != target)
                        continue;

                    long d2 = diag2Pref[r1 + 1][c0] - diag2Pref[r0][c1 + 1];
                    if (d2 != target)
                        continue;

                    boolean ok = true;

                    // Check all row sums
                    for (int r = r0; r <= r1; r++) {
                        long rs = rowPref[r][c1 + 1] - rowPref[r][c0];
                        if (rs != target) {
                            ok = false;
                            break;
                        }
                    }
                    if (!ok)
                        continue;

                    // Check all column sums
                    for (int c = c0; c <= c1; c++) {
                        long cs = colPref[r1 + 1][c] - colPref[r0][c];
                        if (cs != target) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok)
                        return k;
                }
            }
        }

        // If no k>=2 works, any 1x1 is magic
        return 1;
    }

    private boolean existsSquareOfSize(int k, long[][] pre, int m, int n, int threshold) {
        if (k == 0)
            return true;
        for (int i = 0; i + k <= m; i++) {
            for (int j = 0; j + k <= n; j++) {
                long sum = pre[i + k][j + k]
                        - pre[i][j + k]
                        - pre[i + k][j]
                        + pre[i][j];
                if (sum <= threshold)
                    return true;
            }
        }
        return false;
    }
    public int maxSideLength(int[][] mat, int threshold) {
        int m = mat.length, n = mat[0].length;
        // 2D prefix sums: pre[r+1][c+1] = sum of mat[0..r][0..c]
        long[][] pre = new long[m + 1][n + 1];
        for (int r = 0; r < m; r++) {
            long rowRun = 0;
            for (int c = 0; c < n; c++) {
                rowRun += mat[r][c];
                pre[r + 1][c + 1] = pre[r][c + 1] + rowRun;
            }
        }
        int lo = 0, hi = Math.min(m, n);
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2; // upper mid to prevent infinite loop
            if (existsSquareOfSize(mid, pre, m, n, threshold)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    public int[] minBitwiseArray(List<Integer> nums) {
        int mn = Collections.max(nums);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 1; i <= mn; i++) {
            int bo = i | (i + 1);
            map.putIfAbsent(bo, i);
        }
        int[] ans = new int[nums.size()];
        for (int i = 0; i < ans.length; i++) {
            int target = nums.get(i);
            ans[i] = (target == 2) ? -1 : map.getOrDefault(target, -1);
        }
        return ans;
    }

    public int minimumPairRemoval(int[] nums) {
        if (nums.length == 1) {
            return 0;
        }
        int cn = Integer.MIN_VALUE;
        boolean nd = true;
        for (int n : nums) {
            if (n >= cn) {
                cn = n;
            } else {
                nd = false;
                break;
            }
        }
        if (nd) {
            return 0;
        } else {
            int bi = 0;
            int mins = Integer.MAX_VALUE;
            for (int i = 0; i <= nums.length-2; i++) {
                int cs = nums[i]+nums[i+1];
                if (cs < mins) {
                    bi = i;
                    mins = cs;
                }
            }
            int[] res = new int[nums.length-1];
            System.arraycopy(nums, 0, res, 0, bi);
            res[bi] = mins;
            if (res.length - (bi + 1) >= 0)
                System.arraycopy(nums, bi + 2, res, bi + 1, res.length - (bi + 1));
            return 1+minimumPairRemoval(res);
        }
    }

    static class Node {
        long value;
        int left;
        Node prev;
        Node next;

        Node(int value, int left) {
            this.value = value;
            this.left = left;
        }
    }
    static class PQItem implements Comparable<PQItem> {
        Node first;
        Node second;
        long cost;

        PQItem(Node first, Node second, long cost) {
            this.first = first;
            this.second = second;
            this.cost = cost;
        }

        @Override
        public int compareTo(PQItem other) {
            if (this.cost == other.cost) {
                return this.first.left - other.first.left;
            }
            return this.cost < other.cost ? -1 : 1;
        }
    }
    public int minimumPairRemovalII(int[] nums) {
        PriorityQueue<PQItem> pq = new PriorityQueue<>();
        boolean[] merged = new boolean[nums.length];
        int decreaseCount = 0;
        int count = 0;
        Node current = new Node(nums[0], 0);
        for (int i = 1; i < nums.length; i++) {
            Node newNode = new Node(nums[i], i);
            current.next = newNode;
            newNode.prev = current;
            pq.offer(
                    new PQItem(current, newNode, current.value + newNode.value));
            if (nums[i - 1] > nums[i]) {
                decreaseCount++;
            }
            current = newNode;
        }
        while (decreaseCount > 0) {
            PQItem item = pq.poll();
            assert item != null;
            Node first = item.first;
            Node second = item.second;
            long cost = item.cost;
            if (merged[first.left] ||
                    merged[second.left] ||
                    first.value + second.value != cost) {
                continue;
            }
            count++;
            if (first.value > second.value) {
                decreaseCount--;
            }
            Node prevNode = first.prev;
            Node nextNode = second.next;
            first.next = nextNode;
            if (nextNode != null) {
                nextNode.prev = first;
            }
            if (prevNode != null) {
                if (prevNode.value > first.value && prevNode.value <= cost) {
                    decreaseCount--;
                } else if (prevNode.value <= first.value && prevNode.value > cost) {
                    decreaseCount++;
                }
                pq.offer(new PQItem(prevNode, first, prevNode.value + cost));
            }
            if (nextNode != null) {
                if (second.value > nextNode.value && cost <= nextNode.value) {
                    decreaseCount--;
                } else if (second.value <= nextNode.value && cost > nextNode.value) {
                    decreaseCount++;
                }
                pq.offer(new PQItem(first, nextNode, cost + nextNode.value));
            }
            first.value = cost;
            merged[second.left] = true;
        }
        return count;
    }

    public int minPairSum(int[] nums) {
        Arrays.sort(nums);
        int sum = Integer.MIN_VALUE;
        for (int i = 0; i <= nums.length/2-1; i++) {
            int cs = nums[i] + nums[nums.length-1-i];
            sum = Math.max(sum, cs);
        }
        return sum;
    }

    public int minimumDifference(int[] nums, int k) {
        if (k == 1) {
            return 0;
        }
        Arrays.sort(nums);
        int diff = Integer.MAX_VALUE;
        for (int i = 0; i <= nums.length-k; i++) {
            int cd = nums[i+k-1]-nums[i];
            diff = Math.min(cd, diff);
        }
        return diff;
    }

    public List<List<Integer>> minimumAbsDifference(int[] arr) {
        Arrays.sort(arr);
        int diff = Integer.MAX_VALUE;
        for (int i = 0; i <= arr.length-2; i++) {
            int cd = arr[i+1]-arr[i];
            diff = Math.min(diff, cd);
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i <= arr.length-2; i++) {
            int cd = arr[i+1]-arr[i];
            if (cd == diff) {
                List<Integer> list = new ArrayList<>();
                list.add(arr[i]);
                list.add(arr[i+1]);
                res.add(list);
            }
        }
        return res;
    }

    public int minCost(int n, int[][] edges) {
        List<List<int[]>> g = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            g.add(new ArrayList<>());
        }
        for (int[] e : edges) {
            int x = e[0];
            int y = e[1];
            int w = e[2];
            g.get(x).add(new int[] { y, w });
            g.get(y).add(new int[] { x, 2 * w });
        }
        int[] d = new int[n];
        boolean[] visited = new boolean[n];
        Arrays.fill(d, Integer.MAX_VALUE);
        d[0] = 0;
        PriorityQueue<int[]> pq = new PriorityQueue<>(
                Comparator.comparingInt(a -> a[0]));
        pq.offer(new int[] { 0, 0 });
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int dist = current[0];
            int x = current[1];
            if (x == n - 1) {
                return dist;
            }
            if (visited[x]) {
                continue;
            }
            visited[x] = true;
            for (int[] neighbor : g.get(x)) {
                int y = neighbor[0];
                int w = neighbor[1];
                if (dist + w < d[y]) {
                    d[y] = dist + w;
                    pq.offer(new int[] { d[y], y });
                }
            }
        }
        return -1;
    }

    public int minCost(int[][] grid, int k) {
        int m = grid.length;
        int n = grid[0].length;
        List<int[]> points = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                points.add(new int[] { i, j });
            }
        }
        points.sort(Comparator.comparingInt(p -> grid[p[0]][p[1]]));
        int[][] costs = new int[m][n];
        for (int[] row : costs) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        for (int t = 0; t <= k; t++) {
            int minCost = Integer.MAX_VALUE;
            for (int i = 0, j = 0; i < points.size(); i++) {
                minCost = Math.min(
                        minCost,
                        costs[points.get(i)[0]][points.get(i)[1]]);
                if (i + 1 < points.size() &&
                        grid[points.get(i)[0]][points.get(i)[1]] == grid[points.get(i + 1)[0]][points.get(i + 1)[1]]) {
                    continue;
                }
                for (int r = j; r <= i; r++) {
                    costs[points.get(r)[0]][points.get(r)[1]] = minCost;
                }
                j = i + 1;
            }
            for (int i = m - 1; i >= 0; i--) {
                for (int j = n - 1; j >= 0; j--) {
                    if (i == m - 1 && j == n - 1) {
                        costs[i][j] = 0;
                        continue;
                    }
                    if (i != m - 1) {
                        costs[i][j] = Math.min(
                                costs[i][j],
                                costs[i + 1][j] + grid[i + 1][j]);
                    }
                    if (j != n - 1) {
                        costs[i][j] = Math.min(
                                costs[i][j],
                                costs[i][j + 1] + grid[i][j + 1]);
                    }
                }
            }
        }
        return costs[0][0];
    }

    public long minimumCost(String source, String target, char[] original, char[] changed, int[] cost) {
        long[][] paths = new long[26][26];
        for (int i = 0; i <= 25; i++) {
            for (int j = 0; j <= 25; j++) {
                if (i != j) {
                    paths[i][j] = 10000000000L;
                }
            }
        }
        int p = original.length;
        for (int i = 0; i <= p-1; i++) {
            int src = original[i]-'a';
            int dst = changed[i]-'a';
            paths[src][dst] = Math.min(paths[src][dst], cost[i]);
        }
        for (int k = 0; k <= 25; k++) {
            for (int i = 0; i <= 25; i++) {
                for (int j = 0; j <= 25; j++) {
                    if (paths[i][k]+paths[k][j] < paths[i][j]) {
                        paths[i][j] = paths[i][k]+paths[k][j];
                    }
                }
            }
        }
        long tc = 0;
        int l = source.length();
        for (int i = 0; i <= l-1; i++) {
            int src = source.charAt(i)-'a';
            int dst = target.charAt(i)-'a';
            if (paths[src][dst] == 10000000000L) {
                return -1;
            } else {
                tc += paths[src][dst];
            }
        }
        return tc;
    }

    static class Trie {
        Trie[] child = new Trie[26];
        int id = -1;
    }
    private static final int INF = Integer.MAX_VALUE / 2;
    private int add(Trie node, String word, int[] index) {
        for (char ch : word.toCharArray()) {
            int i = ch - 'a';
            if (node.child[i] == null) {
                node.child[i] = new Trie();
            }
            node = node.child[i];
        }
        if (node.id == -1) {
            node.id = ++index[0];
        }
        return node.id;
    }
    public long minimumCost(String source, String target, String[] original, String[] changed, int[] cost) {
        int n = source.length();
        int m = original.length;
        Trie root = new Trie();
        int[] p = { -1 };
        int[][] G = new int[m * 2][m * 2];
        for (int i = 0; i < m * 2; i++) {
            Arrays.fill(G[i], INF);
            G[i][i] = 0;
        }
        for (int i = 0; i < m; i++) {
            int x = add(root, original[i], p);
            int y = add(root, changed[i], p);
            G[x][y] = Math.min(G[x][y], cost[i]);
        }
        int size = p[0] + 1;
        for (int k = 0; k < size; k++) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    G[i][j] = Math.min(G[i][j], G[i][k] + G[k][j]);
                }
            }
        }
        long[] f = new long[n];
        Arrays.fill(f, -1);
        for (int j = 0; j < n; j++) {
            if (j > 0 && f[j - 1] == -1) {
                continue;
            }
            long base = (j == 0 ? 0 : f[j - 1]);
            if (source.charAt(j) == target.charAt(j)) {
                f[j] = f[j] == -1 ? base : Math.min(f[j], base);
            }
            Trie u = root;
            Trie v = root;
            for (int i = j; i < n; i++) {
                u = u.child[source.charAt(i) - 'a'];
                v = v.child[target.charAt(i) - 'a'];
                if (u == null || v == null) {
                    break;
                }
                if (u.id != -1 && v.id != -1 && G[u.id][v.id] != INF) {
                    long newVal = base + G[u.id][v.id];
                    if (f[i] == -1 || newVal < f[i]) {
                        f[i] = newVal;
                    }
                }
            }
        }
        return f[n - 1];
    }

    public char nextGreatestLetter(char[] letters, char target) {
        for (char l : letters) {
            if (l > target) {
                return l;
            }
        }
        return letters[0];
    }

    public int minimumCost(int[] nums) {
        int min1 = Integer.MAX_VALUE;
        int min2 = Integer.MAX_VALUE;
        for (int i = 1; i <= nums.length-1; i++) {
            if (nums[i] < min1) {
                min2 = min1;
                min1 = nums[i];
            } else if (nums[i] < min2) {
                min2 = nums[i];
            }
        }
        return nums[0]+min1+min2;
    }

    private void addToMap(TreeMap<Integer, Integer> map, int x) {
        map.put(x, map.getOrDefault(x, 0) + 1);
    }
    private void removeFromMap(TreeMap<Integer, Integer> map, int x) {
        int cnt = map.getOrDefault(x, 0);
        if (cnt <= 1)
            map.remove(x);
        else
            map.put(x, cnt - 1);
    }
    public long minimumCost(int[] nums, int k, int dist) {
        int n = nums.length;
        long answer = Long.MAX_VALUE;
        long baseCost = nums[0]; // always included
        int need = k - 1; // we must pick need starts from indices 1...n-1
        // Two multiset implemented with TreeMap<value, count>
        TreeMap<Integer, Integer> small = new TreeMap<>(); // holds smallest 'need' values
        TreeMap<Integer, Integer> large = new TreeMap<>(); // holds the rest
        long sumSmall = 0L; // sum of values currently in 'small'
        int smallCount = 0; // number of items in 'small'
        int largeCount = 0; // number of items in 'large'
        int left = 1; // window over indices [left ... right] on the array excluding index 0
        for (int right = 1; right < n; right++) {
            // add nums[right] initially into 'small' (we'll rebalance next)
            addToMap(small, nums[right]);
            sumSmall += nums[right];
            smallCount++;
            // rebalance: ensure smallCount <= need by moving largest from small -> large
            while (smallCount > need) {
                int x = small.lastKey();
                removeFromMap(small, x);
                sumSmall -= x;
                smallCount--;
                addToMap(large, x);
                largeCount++;
            }
            // if smallCount < need, move smallest from large -> small
            while (smallCount < need && largeCount > 0) {
                int x = large.firstKey();
                removeFromMap(large, x);
                largeCount--;
                addToMap(small, x);
                sumSmall += x;
                smallCount++;
            }
            // shrink window from left if it violates the dist constraint
            while (right - left > dist) {
                int val = nums[left];
                // remove from whichever multiset contains it
                if (small.containsKey(val)) {
                    removeFromMap(small, val);
                    sumSmall -= val;
                    smallCount--;
                } else {
                    // must be in large
                    removeFromMap(large, val);
                    largeCount--;
                }
                left++;
                // rebalance after removal
                while (smallCount > need) {
                    int x = small.lastKey();
                    removeFromMap(small, x);
                    sumSmall -= x;
                    smallCount--;

                    addToMap(large, x);
                    largeCount++;
                }
                while (smallCount < need && largeCount > 0) {
                    int x = large.firstKey();
                    removeFromMap(large, x);
                    largeCount--;

                    addToMap(small, x);
                    sumSmall += x;
                    smallCount++;
                }
            }
            // if we currently have exactly 'need' items in small, window is valid candidate
            if (smallCount == need) {
                answer = Math.min(answer, baseCost + sumSmall);
            }
        }
        return answer;
    }

    public boolean isTrionic(int[] nums) {
        int n = nums.length;
        if (n == 3) {
            return false;
        }
        int p = -1;
        int q = -1;
        int prev = nums[0];
        for (int i = 1; i <= n-1; i++) {
            if (p == -1) {
                if (nums[i] > prev) {
                    prev = nums[i];
                } else {
                    if (i == 1 || nums[i] == prev) {
                        return false;
                    } else {
                        p = i-1;
                        prev = nums[i];
                    }
                }
            } else if (q == -1) {
                if (nums[i] < prev) {
                    prev = nums[i];
                } else {
                    if (nums[i] == prev) {
                        return false;
                    } else {
                        q = i-1;
                        prev = nums[i];
                    }
                }
            } else {
                if (nums[i] > prev) {
                    prev = nums[i];
                } else {
                    return false;
                }
            }
        }
        return p != -1 && q != -1;
    }

    public long maxSumTrionic(int[] nums) {
        int n = nums.length;
        long ans = Long.MIN_VALUE;
        for (int i = 0; i < n; i++) {
            int j = i + 1;
            long res = 0;
            // first segment: increasing segment
            while (j < n && nums[j - 1] < nums[j]) {
                j++;
            }
            int p = j - 1;
            if (p == i) {
                continue;
            }
            // second segment: decreasing segment
            res += nums[p] + nums[p - 1];
            while (j < n && nums[j - 1] > nums[j]) {
                res += nums[j];
                j++;
            }
            int q = j - 1;
            if (q == p || q == n - 1 || (j < n && nums[j] <= nums[q])) {
                i = q;
                continue;
            }
            // third segment: increasing segment
            res += nums[q + 1];
            // find the maximum sum of the third segment
            long maxSum = 0;
            long sum = 0;
            for (int k = q + 2; k < n && nums[k] > nums[k - 1]; k++) {
                sum += nums[k];
                maxSum = Math.max(maxSum, sum);
            }
            res += maxSum;
            // find the maximum sum of the first segment
            maxSum = 0;
            sum = 0;
            for (int k = p - 2; k >= i; k--) {
                sum += nums[k];
                maxSum = Math.max(maxSum, sum);
            }
            res += maxSum;
            // update answer
            ans = Math.max(ans, res);
            i = q - 1;
        }
        return ans;
    }

    public int[] constructTransformedArray(int[] nums) {
        int[] result = new int[nums.length];
        for (int i = 0; i <= nums.length-1; i++) {
            if (nums[i] > 0) {
                int rm = nums[i];
                int ri = (rm+i)%nums.length;
                result[i] = nums[ri];
            } else if (nums[i] < 0) {
                int lm = Math.abs(nums[i]);
                int ri = nums.length-1-((nums.length-1-i+lm)%nums.length);
                result[i] = nums[ri];
            } else {
                result[i] = nums[i];
            }
        }
        return result;
    }

    public int minRemoval(int[] nums, int k) {
        int n = nums.length;
        if (n == 1) {
            return 0;
        }
        Arrays.sort(nums);
        int lp = 0;
        int rp = 0;
        int rl = Integer.MIN_VALUE;
        while (rp <= n-1) {
            long multi = (long) nums[lp]*k;
            if (multi >= nums[rp]) {
                rp++;
            } else {
                int cl = rp-lp;
                rl = Math.max(rl, cl);
                lp++;
            }
        }
        if (rl == Integer.MIN_VALUE) {
            return 0;
        } else {
            return n-Math.max(rl, rp-lp);
        }
    }

    public int minimumDeletions(String s) {
        int n = s.length();
        if (n == 1) {
            return 0;
        }
        int na = 0;
        int nb = 0;
        for (char c : s.toCharArray()) {
            if (c == 'a') {
                na += 1;
            } else {
                nb += 1;
            }
        }
        if (na==0 || nb==0) {
            return 0;
        }
        int[] ta = new int[n+1];
        for (int i = 1; i <= n; i++) {
            if (s.charAt(i-1) == 'b') {
                ta[i] = ta[i-1]+1;
            } else {
                ta[i] = ta[i-1];
            }
        }
        int[] tb = new int[n+1];
        for (int i = n-1; i >= 0; i--) {
            if (s.charAt(i) == 'a') {
                tb[i] = tb[i+1]+1;
            } else {
                tb[i] = tb[i+1];
            }
        }
        int cost = Integer.MAX_VALUE;
        for (int i = 0; i <= n; i++) {
            int cc = ta[i]+tb[i];
            cost = Math.min(cost, cc);
        }
        return cost;
    }

    public int longestBalanced(int[] nums) {
        int n = nums.length;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            Set<Integer> seen = new HashSet<>();
            int balance = 0;
            for (int j = i; j < n; j++) {
                if (!seen.contains(nums[j])) {
                    seen.add(nums[j]);
                    if (nums[j] % 2 == 0) {
                        balance++;
                    } else {
                        balance--;
                    }
                }
                if (balance == 0) {
                    ans = Math.max(ans, j - i + 1);
                }
            }
        }
        return ans;
    }

    public int longestBalanced(String s) {
        int n = s.length();
        if (n == 1) {
            return 1;
        }
        int ml = 1;
        for (int i = 0; i < n; i++) {
            int[] freq = new int[26];
            for (int j = i; j < n; j++) {
                freq[s.charAt(j) - 'a']++;
                int target = 0;
                boolean flag = true;
                for (int f : freq) {
                    if (f > 0) {
                        if (target == 0) {
                            target = f;
                        } else {
                            if (f != target) {
                                flag = false;
                                break;
                            }
                        }
                    }
                }
                if (flag) {
                    ml = Math.max(ml, j-i+1);
                }
            }
        }
        return ml;
    }

    private int singlec(String s) {
        int ml = 1;
        char ref = s.charAt(0);
        int cl = 1;
        for (int i = 1; i <= s.length()-1; i++) {
            char cc = s.charAt(i);
            if (cc == ref) {
                cl += 1;
            } else {
                ml = Math.max(ml, cl);
                ref = cc;
                cl = 1;
            }
        }
        return ml;
    }
    private int doublec(char[] c, char x, char y) {
        int n = c.length, max_len = 0;
        int[] first = new int[2 * n + 1];      // → index = diff + n
        Arrays.fill(first, -2);                // -2 means "not set"
        int clear_idx = -1, diff = n;          // diff = 0 + offset
        first[diff] = -1;                     // difference 0 at position -1
        for (int i = 0; i < n; i++) {
            if (c[i] != x && c[i] != y) {     // → forbidden character (the third letter)
                clear_idx = i;                // new segment starts after this position
                diff = n;                    // reset difference to zero
                first[diff] = clear_idx;     // record where this segment starts
            } else {                         // → one of the two letters we care about
                if (c[i] == x) diff++; else diff--;
                if (first[diff] < clear_idx) { // first time we see this diff in current segment
                    first[diff] = i;
                } else {                       // we've seen this diff before – equal counts!
                    max_len = Math.max(max_len, i - first[diff]);
                }
            }
        }
        return max_len;
    }
    private int triplec(String s) {
        Map<String, Integer> map = new HashMap<>();
        int countA = 0, countB = 0, countC = 0;
        int maxLen = 0;
        map.put("0#0", -1);
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == 'a') countA++;
            else if (ch == 'b') countB++;
            else countC++;
            int diffAB = countA - countB;
            int diffAC = countA - countC;
            String key = diffAB + "#" + diffAC;
            if (map.containsKey(key)) {
                maxLen = Math.max(maxLen, i - map.get(key));
            } else {
                map.put(key, i);
            }
        }
        return maxLen;
    }
    public int longestBalancedII(String s) {
        int n = s.length();
        if (n == 1) {
            return 1;
        }
        int one = singlec(s);
        char[] chars = s.toCharArray();
        int two = doublec(chars, 'a', 'b');
        two = Math.max(two, doublec(chars, 'a', 'c'));
        two = Math.max(two, doublec(chars, 'b', 'c'));
        int three = triplec(s);
        return Math.max(Math.max(one, two), three);
    }

    public double champagneTower(int poured, int query_row, int query_glass) {
        double[][] dp = new double[101][101];
        dp[0][0] = poured;
        for (int i = 0; i < query_row; i++) {
            for (int j = 0; j <= i; j++) {
                if (dp[i][j] > 1) {
                    double overflow = (dp[i][j] - 1) / 2.0;
                    dp[i + 1][j] += overflow;
                    dp[i + 1][j + 1] += overflow;
                    dp[i][j] = 1; // cap it
                }
            }
        }
        return Math.min(1, dp[query_row][query_glass]);
    }

    public String addBinary(String a, String b) {
        if (a.length() < b.length()) {
            return addBinary(b, a);
        }
        String res = "";
        int addon = 0;
        for (int i = 0; i <= a.length()-1; i++) {
            int ca = a.charAt(a.length()-1-i)-'0';
            int cb = 0;
            if (i <= b.length()-1) {
                cb = b.charAt(b.length()-1-i)-'0';
            }
            if (ca+cb+addon >= 2) {
                res = String.valueOf(ca+cb+addon-2).concat(res);
                addon = 1;
            } else {
                res = String.valueOf(ca+cb+addon).concat(res);
                addon = 0;
            }
        }
        if (addon == 1) {
            res = "1".concat(res);
        }
        return res;
    }

    public int reverseBits(int n) {
        int res = 0;
        for (int i = 1; i <= 31; i++) {
            int lsb = n&1;
            res = res|lsb;
            res = res<<1;
            n = n>>1;
        }
        return res;
    }

    public List<String> readBinaryWatch(int turnedOn) {
        List<String> result = new ArrayList<>();
        for (int hour = 0; hour < 12; hour++) {
            for (int minute = 0; minute < 60; minute++) {
                // Count total 1-bits in hour and minute
                if (Integer.bitCount(hour) + Integer.bitCount(minute) == turnedOn) {
                    String time = hour + ":" + (minute < 10 ? "0" + minute : minute);
                    result.add(time);
                }
            }
        }
        return result;
    }

    public boolean hasAlternatingBits(int n) {
        int x = n ^ (n >> 1);
        return (x & (x + 1)) == 0;
    }

    public int countBinarySubstrings(String s) {
        int n = s.length();
        if (n == 1) {
            return 0;
        }
        int res = 0;
        char cc = s.charAt(0);
        int cb = 1;
        int pb = 0;
        for (int i = 1; i <= s.length()-1; i++) {
            if (s.charAt(i) == cc) {
                cb += 1;
            } else {
                res += Math.min(pb, cb);
                pb = cb;
                cb = 1;
                cc = s.charAt(i);
            }
        }
        res += Math.min(pb, cb);
        return res;
    }

    public String makeLargestSpecial(String s) {
        if (s.length() <= 2)
            return s;
        List<String> list = new ArrayList<>();
        int count = 0;
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '1') {
                count++;
            } else {
                count--;
            }
            if (count == 0) {
                String inner = s.substring(start + 1, i);
                String processed = "1" + makeLargestSpecial(inner) + "0";
                list.add(processed);
                start = i + 1;
            }
        }
        list.sort(Collections.reverseOrder());
        StringBuilder result = new StringBuilder();
        for (String str : list) {
            result.append(str);
        }
        return result.toString();
    }
}
