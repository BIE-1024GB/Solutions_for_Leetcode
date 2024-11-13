package interview_150;

import org.junit.jupiter.api.Test;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Jiarui BIE
 * @version 1.0
 * @since 2024/6/25
 */
class SolutionTest {
    @Test
    public void testSingleElementArray() {
        int[] nums = {1};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(1, result);
    }

    @Test
    public void testNoDuplicates() {
        int[] nums = {1, 2, 3, 4, 5};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(5, result);
    }

    @Test
    public void testMixedDuplicates() {
        int[] nums = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(10, result); // Each number appears twice
    }

    @Test
    public void testAlternatingNumbers() {
        int[] nums = {1, 2, 1, 2, 1, 2};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(6, result); // Each number appears twice
    }

    @Test
    public void testSingleDuplicate() {
        int[] nums = {1, 1, 2};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(3, result); // Only one duplicate allowed
    }

    @Test
    void majorTest() {
        int[] sequence = new int[] {2, 2, 1, 1, 1, 2, 2};
        Solution solution = new Solution();
        assertEquals(2, solution.majorityElement(sequence));
        int[] seq2 = new int[] {1};
        assertEquals(1, solution.majorityElement(seq2));
    }

    @Test
    void rotateTest() {
        int[] sequence = new int[] {1,2,3,4,5,6,7};
        int[] result = new int[] {5,6,7,1,2,3,4};
        Solution solution = new Solution();
        solution.rotate(sequence, 3);
        for (int i = 0; i <= sequence.length-1; i++) {
            assertEquals(result[i], sequence[i]);
        }
    }

    @Test
    public void testRotateArray() {
        Solution rotateArray = new Solution();
        int[] nums = {1, 2, 3, 4, 5};
        int k = 2;
        int[] expected = {4, 5, 1, 2, 3};
        rotateArray.rotate(nums, k);
        assertArrayEquals(expected, nums);
        int[] n2 = {-33, 22, 6677};
        k = 0;
        expected = new int[] {-33, 22, 6677};
        rotateArray.rotate(n2, k);
        assertArrayEquals(expected, n2);
    }

    @Test
    public void testSell() {
        Solution solution = new Solution();
        int[] prices = new int[] {7, 1, 5, 3, 6, 4};
        assertEquals(5, solution.maxProfit(prices));
        int[] p2 = new int[] {100};
        assertEquals(0, solution.maxProfit(p2));
    }

    @Test
    public void testSell2() {
        Solution solution = new Solution();
        int[] prices = new int[] {7, 1, 5, 3, 6, 4};
        assertEquals(7, solution.maxProfit2(prices));
        int[] p2 = new int[] {1, 2, 3, 4, 5};
        assertEquals(4, solution.maxProfit2(p2));
        int[] p3 = new int[] {115};
        assertEquals(0, solution.maxProfit2(p3));
    }

    @Test
    public void testJump() {
        Solution solution = new Solution();
        int[] p1 = new int[] {0};
        assertTrue(solution.canJump(p1));
        int[] p2 = new int[10000];
        p2[0] = 10000;
        assertTrue(solution.canJump(p2));
        int[] p3 = new int[] {4, 3, 2, 1, 0, 100, 3};
        assertFalse(solution.canJump(p3));
    }

    @Test
    public void testJump2() {
        Solution solution = new Solution();
        int[] p1 = new int[] {2, 3, 0, 1, 4};
        assertEquals(2, solution.jump(p1));
        int[] p2 = new int[] {2, 3, 1, 1, 4};
        assertEquals(2, solution.jump(p2));
    }

    @Test
    public void testHIndex() {
        Solution solution = new Solution();
        int[] p1 = new int[] {3, 0, 6, 1, 5};
        assertEquals(3, solution.hIndex(p1));
    }

    @Test
    public void testRandomSet() {
        Solution.RandomizedSet randomizedSet = new Solution.RandomizedSet();
        assertTrue(randomizedSet.insert(4));
        assertFalse(randomizedSet.insert(4));
        assertFalse(randomizedSet.remove(7));
        assertTrue(randomizedSet.remove(4));
        assertTrue(randomizedSet.insert(4));
        for (int i = 1; i <= 10000; i = i + 1) {
            randomizedSet.insert(i);
        }
        assertTrue(randomizedSet.getRandom() >= 1 && randomizedSet.getRandom() <= 10000);
    }

    @Test
    public void testMultiply() {
        Solution solution = new Solution();
        int[] p1 = new int[] {1, 2, 3, 4};
        int[] a1 = new int[] {24, 12, 8, 6};
        int[] r1 = solution.productExceptSelf(p1);
        for (int i = 0; i <= p1.length-1; i++) {
            assertEquals(a1[i], r1[i]);
        }
    }

    @Test
    public void testGas() {
        Solution solution = new Solution();
        int[] g1 = new int[] {1, 2, 3, 4, 5};
        int[] c1 = new int[] {3, 4, 5, 1, 2};
        assertEquals(3, solution.canCompleteCircuit(g1, c1));
        int[] g2 = new int[] {2, 3, 4};
        int[] c2 = new int[] {3, 4, 3};
        assertEquals(-1, solution.canCompleteCircuit(g2, c2));
    }

    @Test
    public void testCandy() {
        Solution solution = new Solution();
        int[] t1 = new int[] {1, 0, 2};
        int[] t2 = new int[] {1, 2, 2};
        int[] t3 = new int[] {1, 3, 2, 2, 1};
        assertEquals(5, solution.candy(t1));
        assertEquals(4, solution.candy(t2));
        assertEquals(7, solution.candy(t3));
    }

    @Test
    public void testWater() {
        Solution sol = new Solution();
        int[] c0 = new int[] {7, 7};
        assertEquals(0, sol.trap(c0));
        int[] c1 = new int[] {5};
        assertEquals(0, sol.trap(c1));
        int[] c2 = new int[] {4, 3, 1, 1, 2, 0, 5, 1, 5, 3};
        assertEquals(17, sol.trap(c2));
    }

    @Test
    public void testRoman() {
        Solution sol = new Solution();
        String s1 = "III";
        assertEquals(3, sol.romanToInt(s1));
        String s2 = "LVIII";
        assertEquals(58, sol.romanToInt(s2));
        String s3 = "MCMXCIV";
        assertEquals(1994, sol.romanToInt(s3));
    }

    @Test
    public void testIoR() {
        Solution solution = new Solution();
        int t1 = 3749;
        assertEquals("MMMDCCXLIX", solution.intToRoman(t1));
        int t2 = 58;
        assertEquals("LVIII", solution.intToRoman(t2));
        int t3 = 1994;
        assertEquals("MCMXCIV", solution.intToRoman(t3));
    }

    @Test
    public void testZigZag() {
        Solution solution = new Solution();
        assertEquals("PINALSIGYAHRPI", solution.convert("PAYPALISHIRING", 4));
        assertEquals("PHASIYIRPLIGAN", solution.convert("PAYPALISHIRING", 5));
    }

    @Test
    public void testStr() {
        Solution solution = new Solution();
        String h1 = "sadbutsad";
        String n1 = "sad";
        assertEquals(0, solution.strStr(h1, n1));
        String h2 = "leetcode";
        String n2 = "leeto";
        assertEquals(-1, solution.strStr(h2, n2));
    }

    @Test
    public void testJustify() {
        Solution solution = new Solution();
        String[] w1 = new String[] {"This", "is", "an", "example", "of", "text", "justification."};
        int m1 = 16;
        List<String> l1 = new ArrayList<>();
        l1.add("This    is    an");
        l1.add("example  of text");
        l1.add("justification.  ");
        List<String> t1 = solution.fullJustify(w1, m1);
        assertEquals(l1.size(), t1.size());
        for (int i = 0; i <= l1.size()-1; i++) {
            assertEquals(l1.get(i), t1.get(i));
        }
        String[] w2 = new String[] {"What","must","be","acknowledgment","shall","be"};
        int m2 = 16;
        List<String> l2 = new ArrayList<>();
        l2.add("What   must   be");
        l2.add("acknowledgment  ");
        l2.add("shall be        ");
        List<String> t2 = solution.fullJustify(w2, m2);
        assertEquals(l2.size(), t2.size());
        for (int j = 0; j <= l2.size()-1; j++) {
            assertEquals(l2.get(j), t2.get(j));
        }
    }

    @Test
    public void testPalindrome() {
        Solution solution = new Solution();
        String s1 = " ";
        assertTrue(solution.isPalindrome(s1));
        String s2 = "A man, a plan, a canal: Panama";
        assertTrue(solution.isPalindrome(s2));
    }

    @Test
    public void testArea() {
        Solution solution = new Solution();
        int[] h1 = new int[] {1,8,6,2,5,4,8,3,7};
        assertEquals(49, solution.maxArea(h1));
        int[] h2 = new int[] {1,1};
        assertEquals(1, solution.maxArea(h2));
    }

    @Test
    public void testSub() {
        Solution solution = new Solution();
        String s1 = "abcabcbb";
        assertEquals(3, solution.lengthOfLongestSubstring(s1));
        String s2 = "bbbbb";
        assertEquals(1, solution.lengthOfLongestSubstring(s2));
        String s3 = "pwwkew";
        assertEquals(3, solution.lengthOfLongestSubstring(s3));
        String s4 = "au";
        assertEquals(2, solution.lengthOfLongestSubstring(s4));
        String s5 = "dvdf";
        assertEquals(3, solution.lengthOfLongestSubstring(s5));
    }

    @Test
    public void testConcat() {
        Solution solution = new Solution();
        String s1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        String[] w1 = new String[] {"a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a","a"};
        assertEquals(1, solution.findSubstring(s1, w1).size());
        assertEquals(0, solution.findSubstring(s1, w1).getFirst());
    }

    @Test
    public void testMinSub() {
        Solution solution = new Solution();
        String s1 = "ADOBECODEBANC";
        String t1 = "ABC";
        assertEquals("BANC", solution.minWindow(s1, t1));
    }

    @Test
    public void testIso() {
        Solution solution = new Solution();
        String s1 = "egg";
        String t1 = "add";
        assertTrue(solution.isIsomorphic(s1, t1));
        String s2 = "paper";
        String t2 = "title";
        assertTrue(solution.isIsomorphic(s2, t2));
    }

    @Test
    public void testPattern() {
        Solution solution = new Solution();
        String p1 = "abba";
        String s1 = "dog cat cat dog";
        assertTrue(solution.wordPattern(p1, s1));
    }

    @Test
    public void testAna() {
        Solution solution = new Solution();
        String s1 = "anagram";
        String t1 = "nagaram";
        assertTrue(solution.isAnagram(s1, t1));
    }

    @Test
    public void testHappy() {
        Solution solution = new Solution();
        int n1 = 19;
        assertTrue(solution.isHappy(n1));
        int n2 = 2;
        assertFalse(solution.isHappy(n2));
    }

    @Test
    public void testNear() {
        Solution solution = new Solution();
        int[] n1 = new int[] {1, 2, 3, 1};
        int k1 = 3;
        assertTrue(solution.containsNearbyDuplicate(n1, k1));
    }

    @Test
    public void testConsecutive() {
        Solution solution = new Solution();
        int[] n1 = new int[] {100, 4, 200, 3, 1, 2};
        assertEquals(4, solution.longestConsecutive(n1));
    }

    @Test
    public void testArrows() {
        Solution solution = new Solution();
        int[][] p1 = new int[][] {{10, 16}, {2, 8}, {1, 6}, {7, 12}};
        assertEquals(2, solution.findMinArrowShots(p1));
        int[][] p2 = {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
        assertEquals(2, solution.findMinArrowShots(p2));
    }

    @Test
    public void testParentheses() {
        Solution solution = new Solution();
        String s1 = "()";
        assertTrue(solution.isValid(s1));
        String s2 = "(]";
        assertFalse(solution.isValid(s2));
        String s3 = "([)]";
        assertFalse(solution.isValid(s3));
    }

    @Test
    public void testStack() {
        Solution.MinStack minStack = new Solution.MinStack();
        minStack.push(1);
        minStack.push(0);
        minStack.push(1);
        assertEquals(1, minStack.top());
        minStack.pop();
        assertEquals(0, minStack.top());
        assertEquals(0, minStack.getMin());
    }

    @Test
    public void testRPN() {
        Solution solution = new Solution();
        String[] s1 = new String[] {"2","1","+","3","*"};
        assertEquals(9, solution.evalRPN(s1));
        String[] s2 = new String[] {"4","13","5","/","+"};
        assertEquals(6, solution.evalRPN(s2));
    }

    @Test
    public void testCal() {
        Solution solution = new Solution();
        String s1 = " 2-1 + 2 ";
        assertEquals(3, solution.calculate(s1));
        String s2 = "(1+(4+5+2)-3)+(6+8)";
        assertEquals(23, solution.calculate(s2));
        String s3 = "1-(   -2)";
        assertEquals(3, solution.calculate(s3));
    }

    @Test
    public void testLink() {
        Solution.ListNode l1 = new Solution.ListNode(3);
        Solution.ListNode l2 = new Solution.ListNode(2);
        Solution.ListNode l3 = new Solution.ListNode(0);
        Solution.ListNode l4 = new Solution.ListNode(-4);
        l1.next = l2;
        l2.next = l3;
        l3.next = l4;
        l4.next = l2;
        assertTrue(l1.hasCycle(l1));
    }

    @Test
    public void testReveLink() {
        Solution.ListNode listNode1 = new Solution.ListNode(3);
        Solution.ListNode listNode2 = new Solution.ListNode(5);
        listNode1.next = listNode2;
        assertEquals(listNode2, listNode1.reverseBetween(listNode1, 1, 2));
    }

    @Test
    public void testKRev() {
        Solution.ListNode ln1 = new Solution.ListNode(1);
        Solution.ListNode ln2 = new Solution.ListNode(2);
        Solution.ListNode ln3 = new Solution.ListNode(3);
        Solution.ListNode ln4 = new Solution.ListNode(4);
        Solution.ListNode ln5 = new Solution.ListNode(5);
        ln1.next = ln2;
        ln2.next = ln3;
        ln3.next = ln4;
        ln4.next = ln5;
        assertEquals(ln2, ln1.reverseKGroup(ln1, 2));
        assertEquals(ln1, ln2.next);
        assertEquals(ln4, ln1.next);
        assertEquals(ln3, ln4.next);
        assertEquals(ln5, ln3.next);
        assertNull(ln5.next);
    }

    @Test
    public void testDup() {
        Solution.ListNode l1 = new Solution.ListNode(1);
        Solution.ListNode l2 = new Solution.ListNode(1);
        Solution.ListNode l3 = new Solution.ListNode(1);
        Solution.ListNode l4 = new Solution.ListNode(2);
        Solution.ListNode l5 = new Solution.ListNode(3);
        l1.next = l2;
        l2.next = l3;
        l3.next = l4;
        l4.next = l5;
        assertEquals(l4, l1.deleteDuplicates(l1));
        assertEquals(l5, l4.next);
        Solution.ListNode e1 = new Solution.ListNode(1);
        Solution.ListNode e2 = new Solution.ListNode(1);
        e1.next = e2;
        assertNull(e1.deleteDuplicates(e1));
    }

    @Test
    public void testRevLink() {
        Solution.ListNode l1 = new Solution.ListNode(0);
        Solution.ListNode l2 = new Solution.ListNode(1);
        Solution.ListNode l3 = new Solution.ListNode(2);
        l1.next = l2;
        l2.next = l3;
        int k1 = 4;
        assertEquals(l3, l1.rotateRight(l1, k1));
        assertEquals(l1, l3.next);
        assertEquals(l2, l1.next);
        assertNull(l2.next);
    }

    @Test
    public void testPart() {
        Solution.ListNode l1 = new Solution.ListNode(1);
        Solution.ListNode l2 = new Solution.ListNode(2);
        Solution.ListNode l3 = new Solution.ListNode(4);
        Solution.ListNode l4 = new Solution.ListNode(2);
        Solution.ListNode l5 = new Solution.ListNode(3);
        l1.next = l2;
        l2.next = l3;
        l3.next = l4;
        l4.next = l5;
        int x1 = 3;
        assertEquals(l1, l1.partition(l1, x1));
        assertEquals(l2, l1.next);
        assertEquals(l4, l2.next);
        assertEquals(l3, l4.next);
        assertEquals(l5, l3.next);
        assertNull(l5.next);
    }

    @Test
    public void testLRU() {
        Solution.LRUCache cache = new Solution.LRUCache(2);
        cache.put(1, 1);
        cache.put(2, 2);
        assertEquals(1, cache.get(1));
        cache.put(3, 3);
        assertEquals(-1, cache.get(2));
    }

    @Test
    public void testTreeDepth() {
        Solution.TreeNode t1 = new Solution.TreeNode(3, new Solution.TreeNode(9, null, null),
                new Solution.TreeNode(20, new Solution.TreeNode(15, null, null),
                        new Solution.TreeNode(7, null, null)));
        assertEquals(3, t1.maxDepth(t1));
    }

    @Test
    public void testSame() {
        Solution.TreeNode t1 = new Solution.TreeNode(1, new Solution.TreeNode(2, null, null),
                new Solution.TreeNode(3, null, null));
        Solution.TreeNode t2 = new Solution.TreeNode(1, new Solution.TreeNode(2, null, null),
                new Solution.TreeNode(3, null, null));
        assertTrue(t1.isSameTree(t1, t2));
    }
}
