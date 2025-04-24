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
}
