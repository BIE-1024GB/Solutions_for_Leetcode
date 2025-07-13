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
}
