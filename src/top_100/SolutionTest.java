package top_100;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Jiarui BIE
 * @version 1.0
 * @since 2025/07/08
 */
public class SolutionTest {
    @Test
    public void testPhoneNumber() {
        Solution solution = new Solution();
        List<String> exp = new ArrayList<>(Arrays.asList("ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"));
        assertEquals(exp, solution.letterCombinations("23"));
    }

    @Test
    public void testGP() {
        Solution solution = new Solution();
        List<String> exp = new ArrayList<>(Arrays.asList("((()))", "(()())", "(())()", "()(())", "()()()"));
        assertEquals(exp, solution.generateParenthesis(3));
    }

    @Test
    public void testQueens() {
        Solution solution = new Solution();
        List<List<String>> exp = new ArrayList<>();
        List<String> s1 = new ArrayList<>(Arrays.asList(".Q..", "...Q", "Q...", "..Q."));
        List<String> s2 = new ArrayList<>(Arrays.asList("..Q.", "Q...", "...Q", ".Q.."));
        exp.add(s1);
        exp.add(s2);
        assertEquals(exp, solution.solveNQueens(4));
    }

    @Test
    public void testSW() {
        Solution solution = new Solution();
        char[][] board = new char[][] {
                {'A','B','C','E'},
                {'S','F','C','S'},
                {'A','D','E','E'}
        };
        String word = "ABCCED";
        assertTrue(solution.exist(board, word));
    }

    @Test
    public void testMed() {
        Solution solution = new Solution();
        int[] n1 = new int[] {1, 2};
        int[] n2 = new int[] {3, 4};
        assertEquals(2.50000, solution.findMedianSortedArrays(n1, n2));
    }

    @Test
    public void testRS() {
        Solution solution = new Solution();
        int[] nums = new int[] {4,5,6,7,0,1,2};
        assertEquals(4, solution.search(nums, 0));
    }

    @Test
    public void testSR() {
        Solution solution = new Solution();
        int[] nums = new int[] {5, 7, 7, 8, 8, 10};
        int[] exp = new int[] {3, 4};
        assertArrayEquals(exp, solution.searchRange(nums, 8));
    }

    @Test
    public void testSI() {
        Solution solution = new Solution();
        int[] nums = new int[] {1, 3, 5, 6};
        assertEquals(1, solution.searchInsert(nums, 2));
    }

    @Test
    public void testMS() {
        Solution solution = new Solution();
        int[][] matrix = new int[][] {
                {1,3,5,7},
                {10,11,16,20},
                {23,30,34,60}
        };
        assertFalse(solution.searchMatrix(matrix, 13));
    }

    @Test
    public void testBTMPS() {
        Solution.TreeNode n1 = new Solution.TreeNode(-10, new Solution.TreeNode(9), new Solution.TreeNode(20,
                new Solution.TreeNode(15), new Solution.TreeNode(7)));
        assertEquals(42, n1.maxPathSum(n1));
    }

    @Test
    public void testFindMin() {
        Solution solution = new Solution();
        int[] nums = new int[] {5, 1, 2, 3, 4};
        assertEquals(1, solution.findMin(nums));
    }

    @Test
    public void testValidBST() {
        Solution.TreeNode root = new Solution.TreeNode(5, new Solution.TreeNode(4), new Solution.TreeNode(6,
                new Solution.TreeNode(3), new Solution.TreeNode(7)));
        assertFalse(root.isValidBST(root));
    }

    @Test
    public void testMirror() {
        Solution.TreeNode root = new Solution.TreeNode(1, new Solution.TreeNode(2, null, new Solution.TreeNode(3)),
                new Solution.TreeNode(2, null, new Solution.TreeNode(3)));
        assertFalse(root.isSymmetric(root));
    }

    @Test
    public void testMD() {
        Solution.TreeNode root = new Solution.TreeNode(3, new Solution.TreeNode(9), new Solution.TreeNode(20,
                new Solution.TreeNode(15), new Solution.TreeNode(7)));
        assertEquals(3, root.maxDepth(root));
    }

    @Test
    public void testKSTN() {
        Solution.TreeNode root = new Solution.TreeNode(3, new Solution.TreeNode(1, null, new Solution.TreeNode(2)),
                new Solution.TreeNode(4));
        assertEquals(4, root.kthSmallest(root, 4));
    }

    @Test
    public void testLP() {
        Solution solution = new Solution();
        String s = "cbbd";
        assertEquals("bb", solution.longestPalindrome(s));
    }

    @Test
    public void testLVP() {
        Solution solution = new Solution();
        String s = ")()())";
        assertEquals(4, solution.longestValidParentheses(s));
    }

    @Test
    public void testUP() {
        Solution solution = new Solution();
        assertEquals(28, solution.uniquePaths(3, 7));
    }

    @Test
    public void testminPathSum() {
        Solution solution = new Solution();
        int[][] grid = new int[][] {
                {1,3,1},
                {1,5,1},
                {4,2,1}
        };
        assertEquals(7, solution.minPathSum(grid));
    }

    @Test
    public void testStair() {
        Solution solution = new Solution();
        assertEquals(3, solution.climbStairs(3));
    }

    @Test
    public void testEdit() {
        Solution solution = new Solution();
        assertEquals(3, solution.minDistance("horse", "ros"));
    }

    @Test
    public void testWordBreak() {
        Solution solution = new Solution();
        List<String> dict = new ArrayList<>(Arrays.asList("cats", "dog", "sand", "and", "cat"));
        assertFalse(solution.wordBreak("catsandog", dict));
    }

    @Test
    public void testMPS() {
        Solution solution = new Solution();
        int[] nums = new int[] {-2, 0, -1};
        assertEquals(0, solution.maxProduct(nums));
    }

    @Test
    public void testRob() {
        Solution solution = new Solution();
        int[] house = new int[] {2, 7, 9, 3, 1};
        assertEquals(12, solution.rob(house));
    }

    @Test
    public void testSquare() {
        Solution solution = new Solution();
        assertEquals(2, solution.numSquares(9802));
    }

    @Test
    public void testLIS() {
        Solution solution = new Solution();
        int[] nums = new int[] {0,1,0,3,2,3};
        assertEquals(4, solution.lengthOfLIS(nums));
    }

    @Test
    public void testCoins() {
        Solution solution = new Solution();
        int[] coins = new int[] {1, 2, 5};
        assertEquals(3, solution.coinChange(coins, 11));
    }

    @Test
    public void testSum() {
        Solution solution = new Solution();
        int[] nums = new int[] {1, 5, 11, 5};
        assertTrue(solution.canPartition(nums));
    }

    @Test
    public void testLCS() {
        Solution solution = new Solution();
        assertEquals(1, solution.longestCommonSubsequence("psnw", "vozsh"));
    }

    @Test
    public void testIsland() {
        Solution solution = new Solution();
        char[][] grid = new char[][] {
                {'1', '1', '0', '0', '0'},
                {'1', '1', '0', '0', '0'},
                {'0', '0', '1', '0', '0'},
                {'0', '0', '0', '1', '1'},
        };
        assertEquals(3, solution.numIslands(grid));
    }

    @Test
    public void testCourse() {
        Solution solution = new Solution();
        int[][] pre = new int[][] {
                {1, 0},
                {0, 1}
        };
        assertFalse(solution.canFinish(2, pre));
    }

    @Test
    public void testOrange() {
        Solution solution = new Solution();
        int[][] grid = new int[][] {
                {2, 1, 1},
                {1, 1, 0},
                {0, 1, 1}
        };
        assertEquals(4, solution.orangesRotting(grid));
    }

    @Test
    public void testJump() {
        Solution solution = new Solution();
        int[] nums = new int[9999];
        Arrays.fill(nums, 1);
        assertEquals(9998, solution.jump(nums));
    }

    @Test
    public void testCanJump() {
        Solution solution = new Solution();
        int[] nums = new int[] {3, 2, 1, 0, 4};
        assertFalse(solution.canJump(nums));
    }

    @Test
    public void testStock() {
        Solution solution = new Solution();
        int[] price = new int[] {7, 1, 5, 3, 6, 4};
        assertEquals(5, solution.maxProfit(price));
    }

    @Test
    public void testLongconse() {
        Solution solution = new Solution();
        int[] nums = new int[] {0, 1, 2, 4, 8, 5, 6, 7, 9, 3, 55, 88, 77, 99, 999999999};
        assertEquals(10, solution.longestConsecutive(nums));
    }

    @Test
    public void testSubarraySumK() {
        Solution solution = new Solution();
        int[] nums = new int[] {1, 2, 3};
        assertEquals(2, solution.subarraySum(nums, 3));
    }

    @Test
    public void testKthLargest() {
        Solution solution = new Solution();
        int[] nums = new int[] {3, 2, 3, 1, 2, 4, 5, 5, 6};
        assertEquals(4, solution.findKthLargest(nums, 4));
    }

    @Test
    public void testMedianFinder() {
        Solution.MedianFinder medianFinder = new Solution.MedianFinder();
        medianFinder.addNum(1);
        medianFinder.addNum(2);
        assertEquals(1.50000, medianFinder.findMedian());
        medianFinder.addNum(3);
        assertEquals(2.00000, medianFinder.findMedian());
    }

    @Test
    public void testTopKFreq() {
        Solution solution = new Solution();
        int[] nums = new int[] {1, 1, 1, 2, 2, 3};
        int[] exp = {1, 2};
        assertArrayEquals(exp, solution.topKFrequent(nums, 2));
    }

    @Test
    public void testCycle() {
        Solution.ListNode n1 = new Solution.ListNode(3);
        Solution.ListNode n2 = new Solution.ListNode(2);
        Solution.ListNode n3 = new Solution.ListNode(0);
        Solution.ListNode n4 = new Solution.ListNode(-4);
        n1.next = n2;
        n2.next = n3;
        n3.next = n4;
        n4.next = n2;
        assertTrue(n1.hasCycle(n1));
    }

    @Test
    public void testCyclePos() {
        Solution.ListNode n1 = new Solution.ListNode(3);
        Solution.ListNode n2 = new Solution.ListNode(2);
        Solution.ListNode n3 = new Solution.ListNode(0);
        Solution.ListNode n4 = new Solution.ListNode(-4);
        n1.next = n2;
        n2.next = n3;
        n3.next = n4;
        n4.next = n2;
        assertEquals(n2, n1.detectCycle(n1));
    }

    @Test
    public void testSM2() {
        Solution solution = new Solution();
        int[][] matrix = new int[][] {
                {1,4,7,11,15},
                {2,5,8,12,19},
                {3,6,9,16,22},
                {10,13,14,17,24},
                {18,21,23,26,30}
        };
        assertFalse(solution.searchMatrixII(matrix, 20));
    }

    @Test
    public void testLongestSubstring() {
        Solution solution = new Solution();
        String s = "abcabcbb";
        assertEquals(3, solution.lengthOfLongestSubstring(s));
    }

    @Test
    public void testValidParenthesis() {
        Solution solution = new Solution();
        String vp = "([)]";
        assertFalse(solution.isValid(vp));
    }

    @Test
    public void testRectangle() {
        Solution solution = new Solution();
        int[] height = new int[] {2, 1, 5, 6, 2, 3};
        assertEquals(10, solution.largestRectangleArea(height));
    }

    @Test
    public void testRain() {
        Solution solution = new Solution();
        int[] heights = new int[] {7, 6, 5, 4, 3, 2, 1};
        assertEquals(12, solution.maxArea(heights));
    }

    @Test
    public void testTrap() {
        Solution solution = new Solution();
        int[] height = new int[] {0,1,0,2,1,0,1,3,2,1,2,1};
        assertEquals(6, solution.trap(height));
    }

    @Test
    public void testFPI() {
        Solution solution = new Solution();
        int[] num = new int[] {3, 4, -1, 1};
        assertEquals(2, solution.firstMissingPositive(num));
    }
}
