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
}
