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
}
