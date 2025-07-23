package top_100;

import java.util.*;

/**
 * @author Jiarui BIE
 * @version 1.0
 * @since 2025/07/08
 */
public class Solution {
    private void LCdfs(List<String> res, StringBuilder sb, String digits) {
        if (sb.length() == digits.length()) {
            res.add(sb.toString());
        } else {
            char[] letters;
            switch (digits.charAt(sb.length())) {
                case '2' -> letters = new char[] {'a', 'b', 'c'};
                case '3' -> letters = new char[] {'d', 'e', 'f'};
                case '4' -> letters = new char[] {'g', 'h', 'i'};
                case '5' -> letters = new char[] {'j', 'k', 'l'};
                case '6' -> letters = new char[] {'m', 'n', 'o'};
                case '7' -> letters = new char[] {'p', 'q', 'r', 's'};
                case '8' -> letters = new char[] {'t', 'u', 'v'};
                default -> letters = new char[] {'w', 'x', 'y', 'z'};
            }
            for (char c : letters) {
                sb.append(c);
                LCdfs(res, sb, digits);
                sb.deleteCharAt(sb.length()-1);
            }
        }
    }
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (!digits.isEmpty()) {
            LCdfs(res, new StringBuilder(), digits);
        }
        return res;
    }

    private void GPdfs(List<String> res, StringBuilder sb, int lp, int rp, int n) {
        if (sb.length() == n*2) {
            res.add(sb.toString());
        } else {
            if (lp < n) {
                sb.append('(');
                GPdfs(res, sb, lp+1, rp, n);
                sb.deleteCharAt(sb.length()-1);
            }
            if (rp < lp) {
                sb.append(')');
                GPdfs(res, sb, lp, rp+1, n);
                sb.deleteCharAt(sb.length()-1);
            }
        }
    }
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        GPdfs(res, new StringBuilder(), 0, 0, n);
        return res;
    }

    private void CSdfs(int[] candidates, int target, List<List<Integer>> res, List<Integer> curr, int sum, int index) {
        if (sum == target) {
            List<Integer> list = new ArrayList<>(curr);    //don't directly add 'curr'
            res.add(list);
        } else if (sum < target) {
            for (int i = index; i <= candidates.length-1; i++) {
                if (sum+candidates[i] > target) {
                    break;
                } else {
                    curr.add(candidates[i]);
                    CSdfs(candidates, target, res, curr, sum+candidates[i], i);
                    curr.removeLast();
                }
            }
        }
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        CSdfs(candidates, target, res, new ArrayList<>(), 0, 0);
        return res;
    }

    private void Pdfs(int[] nums, List<List<Integer>> res, List<Integer> curr) {
        if (curr.size() == nums.length) {
            List<Integer> list = new ArrayList<>(curr);
            res.add(list);
        } else {
            for (int n : nums) {
                if (!curr.contains(n)) {
                    curr.add(n);
                    Pdfs(nums, res, curr);
                    curr.removeLast();
                }
            }
        }
    }
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Pdfs(nums, res, new ArrayList<>());
        return res;
    }

    private boolean isValid(char[][] board, int row, int col, int n) {
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }
    private List<String> constructSolution(char[][] board) {
        List<String> solution = new ArrayList<>();
        for (char[] row : board) {
            solution.add(new String(row));
        }
        return solution;
    }
    private void backtrack(List<List<String>> solutions, char[][] board, int row, int n) {
        if (row == n) {
            solutions.add(constructSolution(board));
        } else {
            for (int col = 0; col < n; col++) {
                if (isValid(board, row, col, n)) {
                    board[row][col] = 'Q';
                    backtrack(solutions, board, row + 1, n);
                    board[row][col] = '.';
                }
            }
        }
    }
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> solutions = new ArrayList<>();
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = '.';
            }
        }
        backtrack(solutions, board, 0, n);
        return solutions;
    }

    private void Sdfs(List<List<Integer>> res, int[] nums, List<Integer> curr, int index) {
        if (curr.size() == nums.length) {
            List<Integer> list = new ArrayList<>(curr);
            res.add(list);
        } else {
            curr.add(nums[index]);
            List<Integer> list = new ArrayList<>(curr);
            res.add(list);
            int k = 1;
            while (index+k <= nums.length-1) {
                Sdfs(res, nums, curr, index+k);
                k++;
            }
            curr.removeLast();
            if (curr.isEmpty() && index <= nums.length-2) {
                Sdfs(res, nums, curr, index+1);
            }
        }
    }
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        res.add(new ArrayList<>(0));
        Sdfs(res, nums, new ArrayList<>(), 0);
        return res;
    }

    private boolean SWdfs(char[][] board, String word, int index, int r, int c, boolean[][] visit) {
        if (index == word.length()) {
            return true;
        } else {
            if (r-1>=0 && !visit[r-1][c] && board[r-1][c]==word.charAt(index)) {
                visit[r-1][c] = true;
                if (SWdfs(board, word, index+1, r-1, c, visit)) {
                    return true;
                }
                visit[r-1][c] = false;
            }
            if (c+1<=board[0].length-1 && !visit[r][c+1] && board[r][c+1]==word.charAt(index)) {
                visit[r][c+1] = true;
                if (SWdfs(board, word, index+1, r, c+1, visit)) {
                    return true;
                }
                visit[r][c+1] = false;
            }
            if (r+1<=board.length-1 && !visit[r+1][c] && board[r+1][c]==word.charAt(index)) {
                visit[r+1][c] = true;
                if (SWdfs(board, word, index+1, r+1, c, visit)) {
                    return true;
                }
                visit[r+1][c] = false;
            }
            if (c-1>=0 && !visit[r][c-1] && board[r][c-1]==word.charAt(index)) {
                visit[r][c-1] = true;
                if (SWdfs(board, word, index+1, r, c-1, visit)) {
                    return true;
                }
                visit[r][c-1] = false;
            }
            return false;
        }
    }
    public boolean exist(char[][] board, String word) {
        boolean[][] visit = new boolean[board.length][board[0].length];
        for (int i = 0; i <= board.length-1; i++) {
            for (int j = 0; j <= board[0].length-1; j++) {
                if (board[i][j] == word.charAt(0)) {
                    visit[i][j] = true;
                    if (SWdfs(board, word, 1, i, j, visit)) {
                        return true;
                    }
                    visit[i][j] = false;
                }
            }
        }
        return false;
    }

    private boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
    private void backtrack(String s, int start, List<String> current, List<List<String>> result) {
        // If we've reached the end of the string, add the current partition to result
        if (start == s.length()) {
            result.add(new ArrayList<>(current));
            return;
        }
        // Explore all possible partitions
        for (int end = start + 1; end <= s.length(); end++) {
            String substring = s.substring(start, end);
            if (isPalindrome(substring)) {
                current.add(substring);
                backtrack(s, end, current, result);
                current.removeLast();
            }
        }
    }
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        backtrack(s, 0, new ArrayList<>(), result);
        return result;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) {
            return findMedianSortedArrays(nums2, nums1);
        }
        int m = nums1.length;
        int n = nums2.length;
        int left = 0, right = m;
        int halfLen = (m + n + 1) / 2;
        while (left <= right) {
            int partitionX = (left + right) / 2;
            int partitionY = halfLen - partitionX;
            // Handle edge cases where partitions are at the boundaries
            int maxLeftX = (partitionX == 0) ? Integer.MIN_VALUE : nums1[partitionX - 1];
            int minRightX = (partitionX == m) ? Integer.MAX_VALUE : nums1[partitionX];
            int maxLeftY = (partitionY == 0) ? Integer.MIN_VALUE : nums2[partitionY - 1];
            int minRightY = (partitionY == n) ? Integer.MAX_VALUE : nums2[partitionY];
            if (maxLeftX <= minRightY && maxLeftY <= minRightX) {
                // Found the correct partition
                if ((m + n) % 2 == 0) {
                    return (Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY)) / 2.0;
                } else {
                    return Math.max(maxLeftX, maxLeftY);
                }
            } else if (maxLeftX > minRightY) {
                right = partitionX - 1;
            } else {
                left = partitionX + 1;
            }
        }
        throw new IllegalArgumentException("Input arrays are not sorted.");
    }

    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            // Check if the left half is sorted
            if (nums[left] <= nums[mid]) {
                // Target is in the left sorted half
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            // Otherwise, the right half must be sorted
            else {
                // Target is in the right sorted half
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }

    public int[] searchRange(int[] nums, int target) {
        if (nums.length == 0) {
            return new int[] { -1, -1 };
        } else {
            int[] res = new int[] { -1, -1 };
            int lp = 0;
            int rp = nums.length - 1;
            while (lp <= rp) {
                int mid = lp + (rp - lp) / 2;
                if (nums[mid] == target) {
                    if (mid == 0 || nums[mid - 1] < nums[mid]) {
                        res[0] = mid;
                        break;
                    } else {
                        rp = mid - 1;
                    }
                } else {
                    if (nums[mid] < target) {
                        lp = mid + 1;
                    } else {
                        rp = mid - 1;
                    }
                }
            }
            if (res[0] == -1) {
                return res;
            }
            lp = 0;
            rp = nums.length - 1;
            while (lp <= rp) {
                int mid = lp + (rp - lp) / 2;
                if (nums[mid] == target) {
                    if (mid == nums.length - 1 || nums[mid + 1] > nums[mid]) {
                        res[1] = mid;
                        break;
                    } else {
                        lp = mid + 1;
                    }
                } else {
                    if (nums[mid] < target) {
                        lp = mid + 1;
                    } else {
                        rp = mid - 1;
                    }
                }
            }
            return res;
        }
    }

    public int searchInsert(int[] nums, int target) {
        int lp = 0;
        int rp = nums.length-1;
        while (lp <= rp) {
            int mid = lp+(rp-lp)/2;
            if (nums[mid] == target) {
                return mid;
            } else {
                if (nums[mid] > target) {
                    rp = mid-1;
                } else {
                    lp = mid+1;
                }
            }
        }
        return lp;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int lp = 0;
        int rp = rows*cols-1;
        while (lp <= rp) {
            int mid = lp+(rp-lp)/2;
            int mr = mid/cols;
            int mc = mid%cols;
            if (matrix[mr][mc] == target) {
                return true;
            } else {
                if (matrix[mr][mc] < target) {
                    lp = mid+1;
                } else {
                    rp = mid-1;
                }
            }
        }
        return false;
    }

    static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }

        private int maxSum = Integer.MIN_VALUE;
        private int maxGain(TreeNode node) {
            if (node == null)
                return 0;
            // Max sum on the left and right sub-trees of node
            int leftGain = Math.max(maxGain(node.left), 0);
            int rightGain = Math.max(maxGain(node.right), 0);
            // The price to start a new path where 'node' is the highest node
            int priceNewPath = node.val + leftGain + rightGain;
            // Update the global maximum sum
            maxSum = Math.max(maxSum, priceNewPath);
            // Return the maximum contribution this node can make to a path
            return node.val + Math.max(leftGain, rightGain);
        }
        public int maxPathSum(TreeNode root) {
            maxGain(root);
            return maxSum;
        }

        private void inTra(TreeNode root, List<Integer> res) {
            if (root != null) {
                inTra(root.left, res);
                res.add(root.val);
                inTra(root.right, res);
            }
        }
        public List<Integer> inorderTraversal(TreeNode root) {
            List<Integer> res = new ArrayList<>();
            inTra(root, res);
            return res;
        }

        private boolean checker(TreeNode node, Integer low, Integer up) {
            if (node == null) {
                return true;
            } else {
                int val = node.val;

                // Check current node's value against boundaries
                if (low != null && val <= low)
                    return false;
                if (up != null && val >= up)
                    return false;

                // Recursively check left and right subtrees with updated boundaries
                return checker(node.left, low, val) && checker(node.right, val, up);
            }
        }
        public boolean isValidBST(TreeNode root) {
            return checker(root, null, null);
        }

        private boolean isMirror(TreeNode left, TreeNode right) {
            // Base cases
            if (left == null && right == null)
                return true;
            if (left == null || right == null)
                return false;

            // Check values and recurse
            return (left.val == right.val)
                    && isMirror(left.left, right.right)
                    && isMirror(left.right, right.left);
        }
        public boolean isSymmetric(TreeNode root) {
            return isMirror(root.left, root.right);
        }

        public List<List<Integer>> levelOrder(TreeNode root) {
            List<List<Integer>> res = new ArrayList<>();
            if (root != null) {
                Queue<TreeNode> queue = new LinkedList<>();
                queue.add(root);
                while (!queue.isEmpty()) {
                    int ls = queue.size();
                    List<Integer> lr = new ArrayList<>();
                    for (int i = 1; i <= ls; i++) {
                        TreeNode curr = queue.poll();
                        if (curr != null) {
                            lr.add(curr.val);
                            if (curr.left != null) {
                                queue.add(curr.left);
                            }
                            if (curr.right != null) {
                                queue.add(curr.right);
                            }
                        }
                    }
                    res.add(lr);
                }
            }
            return res;
        }

        public int maxDepth(TreeNode root) {
            if (root == null) {
                return 0;
            } else {
                return 1+Math.max(maxDepth(root.left), maxDepth(root.right));
            }
        }

        private int preIndex = 0;
        private final HashMap<Integer, Integer> inorderMap = new HashMap<>();
        private TreeNode buildTreeHelper(int[] preorder, int inStart, int inEnd) {
            if (inStart > inEnd) {
                return null;
            }
            // The current root is the next element in preorder
            int rootValue = preorder[preIndex++];
            TreeNode root = new TreeNode(rootValue);
            // Get the index of the root in inorder to divide left and right subtrees
            int inIndex = inorderMap.get(rootValue);
            // Recursively build left and right subtrees
            root.left = buildTreeHelper(preorder, inStart, inIndex - 1);
            root.right = buildTreeHelper(preorder, inIndex + 1, inEnd);
            return root;
        }
        public TreeNode buildTree(int[] preorder, int[] inorder) {
            // Build a hashmap to store value -> index relations
            for (int i = 0; i < inorder.length; i++) {
                inorderMap.put(inorder[i], i);
            }

            return buildTreeHelper(preorder, 0, inorder.length - 1);
        }

        private TreeNode builder(int[] nums, int lp, int rp) {
            if (lp > rp) {
                return null;
            } else {
                int mid = lp+(rp-lp)/2;
                TreeNode curr = new TreeNode(nums[mid]);
                curr.left = builder(nums, lp, mid-1);
                curr.right = builder(nums, mid+1, rp);
                return curr;
            }
        }
        public TreeNode sortedArrayToBST(int[] nums) {
            return builder(nums, 0, nums.length-1);
        }

        public void flatten(TreeNode root) {
            if (root != null) {
                TreeNode lt = root.left;
                root.left = null;
                flatten(lt);
                TreeNode rt = root.right;
                root.right = lt;
                TreeNode curr = root;
                while (curr.right != null) {
                    curr = curr.right;
                }
                flatten(rt);
                curr.right = rt;
            }
        }

        public List<Integer> rightSideView(TreeNode root) {
            List<Integer> res = new ArrayList<>();
            if (root != null) {
                Queue<TreeNode> queue = new LinkedList<>();
                queue.add(root);
                while (!queue.isEmpty()) {
                    int ls = queue.size();
                    for (int i = 1; i <= ls; i++) {
                        TreeNode curr = queue.poll();
                        if (curr != null) {
                            if (curr.left != null) {
                                queue.add(curr.left);
                            }
                            if (curr.right != null) {
                                queue.add(curr.right);
                            }
                            if (i == ls) {
                                res.add(curr.val);
                            }
                        }
                    }
                }
            }
            return res;
        }

        public TreeNode invertTree(TreeNode root) {
            if (root == null) {
                return null;
            } else {
                TreeNode nl = invertTree(root.right);
                TreeNode nr = invertTree(root.left);
                root.left = nl;
                root.right = nr;
                return root;
            }
        }

        private int count = 0;
        private int result = 0;
        private void inOrder(TreeNode node) {
            if (node == null) return;
            inOrder(node.left);
            count--;
            if (count == 0) {
                result = node.val;
                return;
            }
            inOrder(node.right);
        }
        public int kthSmallest(TreeNode root, int k) {
            count = k;
            inOrder(root);
            return result;
        }

        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            if (root==null || root==p || root==q) {
                return root;
            } else {
                TreeNode lr = lowestCommonAncestor(root.left, p, q);
                TreeNode rr = lowestCommonAncestor(root.right, p, q);
                if (lr!=null && rr!=null) {
                    return root;
                } else {
                    return (lr==null) ? rr : lr;
                }
            }
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

        private int maxDiameter = 0;
        private int depth(TreeNode node) {
            if (node == null) {
                return 0;
            }
            int leftDepth = depth(node.left);
            int rightDepth = depth(node.right);
            maxDiameter = Math.max(maxDiameter, leftDepth + rightDepth);
            return Math.max(leftDepth, rightDepth) + 1;
        }
        public int diameterOfBinaryTree(TreeNode root) {
            depth(root);
            return maxDiameter;
        }
    }

    public int findMin(int[] nums) {
        if (nums.length == 1 || nums[0] < nums[nums.length - 1]) {
            return nums[0];
        } else if (nums[nums.length - 1] < nums[nums.length - 2] && nums[nums.length - 1] < nums[0]) {
            return nums[nums.length - 1];
        } else {
            int lp = 0;
            int rp = nums.length - 1;
            while (lp <= rp) {
                int mid = lp + (rp - lp) / 2;
                if (nums[mid] < nums[mid - 1] && nums[mid] < nums[mid + 1] && nums[mid + 1] < nums[mid - 1]) {
                    return nums[mid];
                } else {
                    if (nums[lp] < nums[rp]) {
                        rp = mid;
                    } else {
                        if (nums[mid] > nums[lp]) {
                            lp = mid;
                        } else {
                            rp = mid;
                        }
                    }
                }
            }
            throw new IllegalArgumentException("Not found");
        }
    }

    public String longestPalindrome(String s) {
        int l = s.length();
        if (l == 1)
            return s;
        boolean[][] dp = new boolean[l][l];
        for (int i = 0; i <= l-1; i++)
            dp[i][i] = true;
        int lpl = 1;
        String lp = s.substring(0, 1);
        for (int i = 0; i <= l-2; i++) {
            if (s.charAt(i) == s.charAt(i+1)) {
                dp[i][i+1] = true;
                if (2 > lpl) {
                    lpl = 2;
                    lp = s.substring(i, i+2);
                }
            }
        }
        for (int sl = 3; sl <= l; sl++) {
            for (int i = 0; i <= l-sl; i++) {
                if (s.charAt(i) == s.charAt(i+sl-1) && dp[i+1][i+sl-2]) {
                    dp[i][i+sl-1] = true;
                    if (sl > lpl) {
                        lpl = sl;
                        lp = s.substring(i, i+sl);
                    }
                }
            }
        }
        return lp;
    }

    public int longestValidParentheses(String s) {
        int maxLen = 0;
        int n = s.length();
        int[] dp = new int[n];
        for (int i = 1; i < n; i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else {
                    if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                        dp[i] = dp[i - 1] + 2 + ((i - dp[i - 1] - 2) >= 0 ? dp[i - dp[i - 1] - 2] : 0);
                    }
                }
                maxLen = Math.max(maxLen, dp[i]);
            }
        }
        return maxLen;
    }

    public int uniquePaths(int m, int n) {
        if (m==1 || n==1) {
            return 1;
        } else {
            int[][] dp = new int[m][n];
            for (int i = 0; i <= n-1; i++) {
                dp[0][i] = 1;
            }
            for (int j = 0; j <= m-1; j++) {
                dp[j][0] = 1;
            }
            for (int p = 1; p <= m-1; p++) {
                for (int q = 1; q <= n-1; q++) {
                    dp[p][q] = dp[p-1][q]+dp[p][q-1];
                }
            }
            return dp[m-1][n-1];
        }
    }

    public int minPathSum(int[][] grid) {
        if (grid.length == 1 && grid[0].length == 1) {
            return grid[0][0];
        } else {
            int[][] dp = new int[grid.length][grid[0].length];
            dp[0][0] = grid[0][0];
            for (int i = 1; i <= grid[0].length-1; i++) {
                dp[0][i] = dp[0][i-1]+grid[0][i];
            }
            for (int i = 1; i <= grid.length-1; i++) {
                dp[i][0] = dp[i-1][0]+grid[i][0];
            }
            for (int m = 1; m <= grid.length-1; m++) {
                for (int n = 1; n <= grid[0].length-1; n++) {
                    dp[m][n] = Math.min(dp[m-1][n], dp[m][n-1])+grid[m][n];
                }
            }
            return dp[dp.length-1][dp[0].length-1];
        }
    }

    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        } else {
            int[] dp = new int[n+1];
            dp[0] = 1;
            dp[1] = 1;
            for (int i = 2; i <= n; i++) {
                dp[i] = dp[i-1]+dp[i-2];
            }
            return dp[n];
        }
    }

    public int minDistance(String word1, String word2) {
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        dp[0][0] = 0;
        for (int i = 1; i <= word2.length(); i++) {
            dp[0][i] = i;
        }
        for (int i = 1; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], Math.min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
                }
            }
        }
        return dp[word1.length()][word2.length()];
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> r1 = new ArrayList<>();
        r1.add(1);
        res.add(r1);
        if (numRows > 1) {
            List<Integer> r2 = new ArrayList<>();
            r2.add(1);
            r2.add(1);
            res.add(r2);
            if (numRows > 2) {
                for (int i = 3; i <= numRows; i++) {
                    List<Integer> curr = new ArrayList<>();
                    List<Integer> prev = res.get(i-2);
                    for (int k = 1; k <= i; k++) {
                        if (k == 1 || k == i) {
                            curr.add(1);
                        } else {
                            curr.add(prev.get(k-2)+prev.get(k-1));
                        }
                    }
                    res.add(curr);
                }
            }
        }
        return res;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        HashSet<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= dp.length - 1; i++) {
            for (int j = 0; j <= i - 1; j++) {
                if (dp[j]) {
                    if (set.contains(s.substring(j, i))) {
                        dp[i] = true;
                        break;
                    }
                }
            }
        }
        return dp[dp.length - 1];
    }

    public int maxProduct(int[] nums) {
        int max_so_far = nums[0];
        int min_so_far = nums[0];
        int result = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int current = nums[i];
            int temp_max = Math.max(current, Math.max(max_so_far * current, min_so_far * current));
            min_so_far = Math.min(current, Math.min(max_so_far * current, min_so_far * current));
            max_so_far = temp_max;
            result = Math.max(result, max_so_far);
        }
        return result;
    }

    public int rob(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        } else {
            int[] dp = new int[nums.length+1];
            dp[0] = 0;
            dp[1] = nums[0];
            for (int i = 2; i <= nums.length; i++) {
                dp[i] = Math.max(dp[i-1], nums[i-1]+dp[i-2]);
            }
            return dp[dp.length-1];
        }
    }

    public int numSquares(int n) {
        // note: more efficient solution available
        if (n == 1) {
            return 1;
        } else {
            int[] dp = new int[n];
            int ps = 1;
            for (int i = 0; i <= n-1; i++) {
                if (i+1 == ps*ps) {
                    dp[i] = 1;
                    ps += 1;
                } else {
                    int mn = Integer.MAX_VALUE;
                    for (int j = 0; j <= i-1; j++) {
                        int diff = i-j;
                        mn = Math.min(mn, dp[j]+dp[diff-1]);
                    }
                    dp[i] = mn;
                }
            }
            return dp[n-1];
        }
    }
}
