package miscellaneous;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class SolutionTest {
    @Test
    public void testLang() {
        Solution solution = new Solution();
        int n = 3;
        int[][] languages = new int[][] {
                {2},
                {1, 3},
                {1, 2},
                {3}
        };
        int[][] fri = new int[][] {
                {1, 4},
                {1, 2},
                {3, 4},
                {2, 3}
        };
        assertEquals(2, solution.minimumTeachings(n, languages, fri));
    }

    @Test
    public void testRI() {
        Solution solution = new Solution();
        int prev = -123;
        assertEquals(-321, solution.reverse(prev));
    }

    @Test
    public void testVer() {
        Solution solution = new Solution();
        String v1 = "1.2";
        String v2 = "1.10";
        assertEquals(-1, solution.compareVersion(v1, v2));
    }

    @Test
    public void testFraction() {
        Solution solution = new Solution();
        assertEquals("0.2(095238)", solution.fractionToDecimal(22, 105));
    }

    @Test
    public void testTP() {
        Solution solution = new Solution();
        int[] sides = new int[] {1,2,1,10};
        assertEquals(0, solution.largestPerimeter(sides));
    }

    @Test
    public void testATOI() {
        Solution solution = new Solution();
        String str = "  0000000000012345678";
        assertEquals(12345678, solution.myAtoi(str));
    }

    @Test
    public void testTriangulation() {
        Solution solution = new Solution();
        int[] vertex = new int[] {1,3,1,4,1,5};
        assertEquals(13, solution.minScoreTriangulation(vertex));
    }

    @Test
    public void testTS() {
        Solution solution = new Solution();
        int[] arr = new int[] {1, 2, 3, 4, 5};
        assertEquals(8, solution.triangularSum(arr));
    }
}
