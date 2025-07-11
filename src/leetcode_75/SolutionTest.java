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
}
