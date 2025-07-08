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
}
