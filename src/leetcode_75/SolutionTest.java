package leetcode_75;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Jiarui BIE
 * @version 1.0
 * @since 2025/04/24
 */
public class SolutionTest {
    @Test
    public void testAltMerge() {
        Solution solution = new Solution();
        String w1 = "abc";
        String w2 = "pqr";
        assertEquals("apbqcr", solution.mergeAlternately(w1, w2));
    }

    @Test
    public void testStringGCD() {
        Solution solution = new Solution();
        String s1 = "ABCABC";
        String s2 = "ABC";
        assertEquals("ABC", solution.gcdOfStrings(s1, s2));
    }

    @Test
    public void testFlower() {
        Solution solution = new Solution();
        int[] f1 = new int[] {0};
        assertTrue(solution.canPlaceFlowers(f1, 1));
        int[] f2 = new int[] {1, 0, 0, 0, 1};
        assertFalse(solution.canPlaceFlowers(f2, 2));
    }

    @Test
    public void testRW() {
        Solution solution = new Solution();
        String s1 = "a good   example";
        assertEquals("example good a", solution.reverseWords(s1));
    }

    @Test
    public void testIT() {
        Solution solution = new Solution();
        int[] n1 = new int[] {1, 2, 3, 4, 5};
        assertTrue(solution.increasingTriplet(n1));
        int[] n2 = new int[] {5, 4, 3, 1, 2};
        assertFalse(solution.increasingTriplet(n2));
    }

    @Test
    public void testCompress() {
        Solution solution = new Solution();
        char[] c1 = new char[] {'a', 'b', 'c'};
        assertEquals(3, solution.compress(c1));
        char[] c2 = new char[] {'a', 'a', 'b', 'b', 'c', 'c', 'c'};
        assertEquals(6, solution.compress(c2));
        assertEquals('a', c2[0]);
        assertEquals('2', c2[1]);
        assertEquals('b', c2[2]);
        assertEquals('2', c2[3]);
        assertEquals('c', c2[4]);
        assertEquals('3', c2[5]);
    }

    @Test
    public void testSub() {
        Solution solution = new Solution();
        String s = "axc";
        String t = "ahbgdc";
        assertFalse(solution.isSubsequence(s, t));
    }

    @Test
    public void testArea() {
        Solution solution = new Solution();
        int[] height = new int[] {1,8,6,2,5,4,8,3,7};
        assertEquals(49, solution.maxArea(height));
    }

    @Test
    public void testOperation() {
        Solution solution = new Solution();
        int[] n1 = new int[] {4,4,1,3,1,3,2,2,5,5,1,5,2,1,2,3,5,4};
        assertEquals(2, solution.maxOperations(n1, 2));
    }

    @Test
    public void testMaxAvg() {
        Solution solution = new Solution();
        int[] n1 = new int[] {1,12,-5,-6,50,3};
        assertEquals(12.75000, solution.findMaxAverage(n1, 4));
    }

    @Test
    public void testMaxVowel() {
        Solution solution = new Solution();
        String s = "abciiidef";
        assertEquals(3, solution.maxVowels(s, 3));
    }

    @Test
    public void testLO() {
        Solution solution = new Solution();
        int[] n = new int[] {1,1,1,0,0,0,1,1,1,1,0};
        assertEquals(6, solution.longestOnes(n, 2));
    }

    @Test
    public void testDO() {
        Solution solution = new Solution();
        int[] n = new int[] {0,1,1,1,0,1,1,0,1};
        assertEquals(5, solution.longestSubarray(n));
    }

    @Test
    public void testAlti() {
        Solution solution = new Solution();
        int[] g = new int[] {-4,-3,-2,-1,4,3,2};
        assertEquals(0, solution.largestAltitude(g));
    }

    @Test
    public void testPivot() {
        Solution solution = new Solution();
        int[] n = new int[] {2,1,-1};
        assertEquals(0, solution.pivotIndex(n));
    }

    @Test
    public void testOccur() {
        Solution solution = new Solution();
        int[] arr = new int[] {-3,0,1,-3,1,1,1,-3,10,0};
        assertTrue(solution.uniqueOccurrences(arr));
    }

    @Test
    public void testClose() {
        Solution solution = new Solution();
        String w1 = "cabbba";
        String w2 = "abbccc";
        assertTrue(solution.closeStrings(w1, w2));
    }

    @Test
    public void testEP() {
        Solution solution = new Solution();
        int[][] grid = new int[][] {
                {3,1,2,2},
                {1,4,4,5},
                {2,4,2,2},
                {2,4,2,2}
        };
        assertEquals(3, solution.equalPairs(grid));
    }

    @Test
    public void testRS() {
        Solution solution = new Solution();
        String s = "leet**cod*e";
        assertEquals("lecoe", solution.removeStars(s));
    }

    @Test
    public void testDecode() {
        Solution solution = new Solution();
        String s = "3[a2[c]]";
        assertEquals("accaccacc", solution.decodeString(s));
    }

    @Test
    public void testRecent() {
        Solution.RecentCounter recentCounter = new Solution.RecentCounter();
        assertEquals(1, recentCounter.ping(1));
        assertEquals(2, recentCounter.ping(100));
        assertEquals(3, recentCounter.ping(3001));
        assertEquals(3, recentCounter.ping(3002));
    }

    @Test
    public void testDota() {
        Solution solution = new Solution();
        assertEquals("Dire", solution.predictPartyVictory("RDD"));
    }

    @Test
    public void testPairSum() {
        Solution.ListNode n1 = new Solution.ListNode(5);
        Solution.ListNode n2 = new Solution.ListNode(4);
        Solution.ListNode n3 = new Solution.ListNode(2);
        Solution.ListNode n4 = new Solution.ListNode(1);
        n1.next = n2;
        n2.next = n3;
        n3.next = n4;
        assertEquals(6, n1.pairSum(n1));
    }

    @Test
    public void testMD() {
        Solution.TreeNode n1 = new Solution.TreeNode(1);
        Solution.TreeNode n2 = new Solution.TreeNode(2);
        n1.right = n2;
        assertEquals(2, n1.maxDepth(n1));
    }

    @Test
    public void testGN() {
        Solution.TreeNode n1 = new Solution.TreeNode(3);
        Solution.TreeNode n2 = new Solution.TreeNode(3);
        Solution.TreeNode n3 = new Solution.TreeNode(4);
        Solution.TreeNode n4 = new Solution.TreeNode(2);
        n1.left = n2;
        n2.left = n3;
        n2.right = n4;
        assertEquals(3, n1.goodNodes(n1));
    }

    @Test
    public void testPS() {
        Solution.TreeNode n1 = new Solution.TreeNode(10);
        Solution.TreeNode n2 = new Solution.TreeNode(5);
        Solution.TreeNode n3 = new Solution.TreeNode(-3);
        Solution.TreeNode n4 = new Solution.TreeNode(3);
        Solution.TreeNode n5 = new Solution.TreeNode(2);
        Solution.TreeNode n6 = new Solution.TreeNode(11);
        Solution.TreeNode n7 = new Solution.TreeNode(3);
        Solution.TreeNode n8 = new Solution.TreeNode(-2);
        Solution.TreeNode n9 = new Solution.TreeNode(1);
        n1.left = n2;
        n1.right = n3;
        n2.left = n4;
        n2.right = n5;
        n3.right = n6;
        n4.left = n7;
        n4.right = n8;
        n5.right = n9;
        assertEquals(3, n1.pathSum(n1, 8));
    }

    @Test
    public void testMLS() {
        Solution.TreeNode n1 = new Solution.TreeNode(1);
        Solution.TreeNode n2 = new Solution.TreeNode(7);
        Solution.TreeNode n3 = new Solution.TreeNode(0);
        Solution.TreeNode n4 = new Solution.TreeNode(7);
        Solution.TreeNode n5 = new Solution.TreeNode(-8);
        n1.left = n2;
        n1.right = n3;
        n2.left = n4;
        n2.right = n5;
        assertEquals(2, n1.maxLevelSum(n1));
    }
}
