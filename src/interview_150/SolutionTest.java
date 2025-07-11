package interview_150;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * @author Jiarui BIE
 * @version 1.11
 * @since 2024/06/25
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
    void testMajor() {
        int[] sequence = new int[] {2, 2, 1, 1, 1, 2, 2};
        Solution solution = new Solution();
        assertEquals(2, solution.majorityElement(sequence));
        int[] seq2 = new int[] {1};
        assertEquals(1, solution.majorityElement(seq2));
    }

    @Test
    void testRotate() {
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
        e1.next = new Solution.ListNode(1);
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

    @Test
    public void testMirror() {
        Solution.TreeNode t1 = new Solution.TreeNode(1, new Solution.TreeNode(2, new Solution.TreeNode(3), new Solution.TreeNode(4)),
                new Solution.TreeNode(2, new Solution.TreeNode(4), new Solution.TreeNode(3)));
        assertTrue(t1.isSymmetric(t1));
    }

    @Test
    public void testSum() {
        Solution.TreeNode n1 = new Solution.TreeNode(1, new Solution.TreeNode(2), new Solution.TreeNode(3));
        assertFalse(n1.hasPathSum(n1, 5));
        Solution.TreeNode n2 = new Solution.TreeNode(1, new Solution.TreeNode(2), null);
        assertFalse(n2.hasPathSum(n2, 1));
    }

    @Test
    public void testAllSum() {
        Solution.TreeNode n1 = new Solution.TreeNode(1, new Solution.TreeNode(2), new Solution.TreeNode(3));
        assertEquals(25, n1.sumNumbers(n1));
    }

    @Test
    public void testMAXSum() {
        Solution.TreeNode n1 = new Solution.TreeNode(-10, new Solution.TreeNode(9),
                new Solution.TreeNode(20, new Solution.TreeNode(15), new Solution.TreeNode(7)));
        assertEquals(42, n1.maxPathSum(n1));
    }

    @Test
    public void testCount() {
        Solution.TreeNode n1 = new Solution.TreeNode(1, new Solution.TreeNode(2, new Solution.TreeNode(4),
                new Solution.TreeNode(5)), new Solution.TreeNode(3, new Solution.TreeNode(6), null));
        assertEquals(6, n1.countNodes(n1));
    }

    @Test
    public void testLCA() {
        Solution.TreeNode n1 = new Solution.TreeNode(3);
        Solution.TreeNode n2 = new Solution.TreeNode(5);
        Solution.TreeNode n3 = new Solution.TreeNode(1);
        Solution.TreeNode n4 = new Solution.TreeNode(6);
        Solution.TreeNode n5 = new Solution.TreeNode(2);
        Solution.TreeNode n6 = new Solution.TreeNode(0);
        Solution.TreeNode n7 = new Solution.TreeNode(8);
        Solution.TreeNode n8 = new Solution.TreeNode(7);
        Solution.TreeNode n9 = new Solution.TreeNode(4);
        n1.left = n2;
        n1.right = n3;
        n2.left = n4;
        n2.right = n5;
        n3.left = n6;
        n3.right = n7;
        n5.left = n8;
        n5.right = n9;
        assertEquals(n2, n1.lowestCommonAncestor(n1, n2, n9));
    }

    @Test
    public void testRight() {
        Solution.TreeNode n1 = new Solution.TreeNode(1, new Solution.TreeNode(2, null, new Solution.TreeNode(5)),
                new Solution.TreeNode(3, null, new Solution.TreeNode(4)));
        List<Integer> re = n1.rightSideView(n1);
        List<Integer> tr = new ArrayList<>();
        tr.add(1);
        tr.add(3);
        tr.add(4);
        for (int i = 0; i <= tr.size()-1; i++) {
            assertEquals(tr.get(i).intValue(), re.get(i).intValue());
        }
    }

    @Test
    public void testAvg() {
        Solution.TreeNode n1 = new Solution.TreeNode(3, new Solution.TreeNode(9,
                new Solution.TreeNode(15), new Solution.TreeNode(7)), new Solution.TreeNode(20));
        List<Double> exp = new ArrayList<>();
        exp.add(3.00000);
        exp.add(14.50000);
        exp.add(11.00000);
        List<Double> re = n1.averageOfLevels(n1);
        for (int i = 0; i <= exp.size()-1; i++) {
            assertEquals(exp.get(i).doubleValue(), re.get(i).doubleValue());
        }
    }

    @Test
    public void testMinDiff() {
        Solution.TreeNode n1 = new Solution.TreeNode(4, new Solution.TreeNode(2, new Solution.TreeNode(1),
                new Solution.TreeNode(3)), new Solution.TreeNode(6));
        assertEquals(1, n1.getMinimumDifference(n1));
    }

    @Test
    public void testKmin() {
        Solution.TreeNode n1 = new Solution.TreeNode(5, new Solution.TreeNode(3, new Solution.TreeNode(2,
                new Solution.TreeNode(1), null), new Solution.TreeNode(4)), new Solution.TreeNode(6));
        assertEquals(3, n1.kthSmallest(n1, 3));
    }

    @Test
    public void testBST() {
        Solution.TreeNode n1 = new Solution.TreeNode(2, new Solution.TreeNode(1, new Solution.TreeNode(0),
                new Solution.TreeNode(4)), new Solution.TreeNode(3));
        assertFalse(n1.isValidBST(n1));
    }

    @Test
    public void testIsland() {
        Solution solution = new Solution();
        char[][] grid = {{'1', '1', '0', '0', '0'},
                            {'1', '1', '0', '0', '0'},
                            {'0', '0', '1', '0', '0'},
                            {'0', '0', '0', '1', '1'}};
        assertEquals(3, solution.numIslands(grid));
    }

    @Test
    public void testCalDiv() {
        Solution solution = new Solution();
        List<String> e1 = new ArrayList<>();
        e1.add("a");
        e1.add("b");
        List<String> e2 = new ArrayList<>();
        e2.add("b");
        e2.add("c");
        List<List<String>> equations = new ArrayList<>();
        equations.add(e1);
        equations.add(e2);
        double[] vals = {2.0, 3.0};
        List<List<String>> queries = getLists();
        double[] exp = {6.00000, 0.50000, -1.00000, 1.00000, -1.00000};
        double[] res = solution.calcEquation(equations, vals, queries);
        for (int i = 0; i <= exp.length-1; i++) {
            assertEquals(exp[i], res[i]);
        }
    }
    private static List<List<String>> getLists() {
        List<String> q1 = new ArrayList<>();
        q1.add("a");
        q1.add("c");
        List<String> q2 = new ArrayList<>();
        q2.add("b");
        q2.add("a");
        List<String> q3 = new ArrayList<>();
        q3.add("a");
        q3.add("e");
        List<String> q4 = new ArrayList<>();
        q4.add("a");
        q4.add("a");
        List<String> q5 = new ArrayList<>();
        q5.add("x");
        q5.add("x");
        List<List<String>> queries = new ArrayList<>();
        queries.add(q1);
        queries.add(q2);
        queries.add(q3);
        queries.add(q4);
        queries.add(q5);
        return queries;
    }

    @Test
    public void testCourse() {
        Solution solution = new Solution();
        int numCourses = 2;
        int[][] prerequisites = {{1, 0}};
        assertTrue(solution.canFinish(numCourses, prerequisites));
        prerequisites = new int[][]{{1, 0}, {0, 1}};
        assertFalse(solution.canFinish(numCourses, prerequisites));
    }

    @Test
    public void testSaL() {
        Solution solution = new Solution();
        int[][] board = {
                {-1, -1, -1, -1, -1, -1},
                {-1, -1, -1, -1, -1, -1},
                {-1, -1, -1, -1, -1, -1},
                {-1, 35, -1, -1, 13, -1},
                {-1, -1, -1, -1, -1, -1},
                {-1, 15, -1, -1, -1, -1}
        };
        assertEquals(4, solution.snakesAndLadders(board));
    }

    @Test
    public void testGene() {
        Solution solution = new Solution();
        String s1 = "AACCGGTT";
        String e1 = "AACCGGTA";
        String[] b1 = new String[] {"AACCGGTA"};
        assertEquals(1, solution.minMutation(s1, e1, b1));
    }

    @Test
    public void testWord() {
        Solution solution = new Solution();
        String s1 = "hit";
        String e1 = "cog";
        String[] d1 = new String[] {"hot", "dot", "dog", "lot", "log", "cog"};
        List<String> l1 = Arrays.asList(d1);
        assertEquals(5, solution.ladderLength(s1, e1, l1));
    }

    @Test
    public void testWordDict() {
        Solution.WordDictionary wordDictionary = new Solution.WordDictionary();
        wordDictionary.addWord("aid");
        wordDictionary.addWord("mad");
        wordDictionary.addWord("misao");
        assertTrue(wordDictionary.search("a.."));
        assertFalse(wordDictionary.search("m.p"));
        assertTrue(wordDictionary.search("m.s.o"));
    }

    @Test
    public void testQueen() {
        Solution solution = new Solution();
        assertEquals(2, solution.totalNQueens(4));
        assertEquals(1, solution.totalNQueens(1));
        assertEquals(0, solution.totalNQueens(2));
    }

    @Test
    public void testWS() {
        Solution solution = new Solution();
        char[][] b1 = new char[][] {{'A', 'B', 'C', 'E'},
                {'S', 'F', 'C', 'S'},
                {'A', 'D', 'E', 'E'}};
        String w1 = "ABCCED";
        assertTrue(solution.exist(b1, w1));
    }

    @Test
    public void testSort() {
        Solution.ListNode l1 = new Solution.ListNode(4);
        Solution.ListNode l2 = new Solution.ListNode(2);
        Solution.ListNode l3 = new Solution.ListNode(1);
        Solution.ListNode l4 = new Solution.ListNode(3);
        l1.next = l2;
        l2.next = l3;
        l3.next = l4;
        Solution.ListNode rl = l1.sortList(l1);
        assertEquals(1, rl.val);
        assertEquals(2, rl.next.val);
        assertEquals(3, rl.next.next.val);
        assertEquals(4, rl.next.next.next.val);
    }

    @Test
    public void testCircKadane() {
        Solution solution = new Solution();
        int[] n1 = new int[]{5, 5, 3};
        assertEquals(13, solution.maxSubarraySumCircular(n1));
        int[] n2 = new int[]{5, -3, 5};
        assertEquals(10, solution.maxSubarraySumCircular(n2));
        int[] n3 = new int[]{-3, -9, -6};
        assertEquals(-3, solution.maxSubarraySumCircular(n3));
    }

    @Test
    public void testBS() {
        Solution solution = new Solution();
        int[] n1 = new int[] {1, 3, 5, 6};
        int t1 = 5;
        int t2 = 2;
        int t3 = 7;
        int t4 = 1;
        int t5 = 6;
        assertEquals(2, solution.searchInsert(n1, t1));
        assertEquals(1, solution.searchInsert(n1, t2));
        assertEquals(4, solution.searchInsert(n1, t3));
        assertEquals(0, solution.searchInsert(n1, t4));
        assertEquals(3, solution.searchInsert(n1, t5));
    }

    @Test
    public void testMin() {
        Solution solution = new Solution();
        int[] n1 = new int[] {5, 1, 2, 3, 4};
        assertEquals(1, solution.findMin(n1));
    }

    @Test
    public void testMedian() {
        Solution solution = new Solution();
        int[] n1 = new int[]{1, 3};
        int[] n2 = new int[]{2};
        assertEquals(2.00000, solution.findMedianSortedArrays(n1, n2));
        int[] n3 = new int[]{1, 3};
        int[] n4 = new int[]{2, 4};
        assertEquals(2.50000, solution.findMedianSortedArrays(n3, n4));
    }

    @Test
    public void testMinHeap() {
        Solution solution = new Solution();
        int[] n1 = new int[]{3, 2, 1, 5, 6, 4};
        assertEquals(5, solution.findKthLargest(n1, 2));
    }

    @Test
    public void testIPO() {
        Solution solution = new Solution();
        int[] p1 = new int[]{1, 2, 3};
        int[] c1 = new int[]{0, 1, 2};
        assertEquals(6, solution.findMaximizedCapital(3, 0, p1, c1));
    }

    @Test
    public void testMed() {
        Solution.MedianFinder medianFinder = new Solution.MedianFinder();
        medianFinder.addNum(-1);
        medianFinder.addNum(-2);
        medianFinder.addNum(-3);
        assertEquals(-2.00000, medianFinder.findMedian());
        medianFinder.addNum(-4);
        assertEquals(-2.50000, medianFinder.findMedian());
        medianFinder.addNum(-5);
        assertEquals(-3.00000, medianFinder.findMedian());
    }

    @Test
    public void testBA() {
        Solution solution = new Solution();
        String a1 = "11";
        String b1 = "1";
        assertEquals("100", solution.addBinary(a1, b1));
    }

    @Test
    public void testBR() {
        Solution solution = new Solution();
        int n1 = 0b00000010100101000001111010011100;
        assertEquals(964176192, solution.reverseBits(n1));
    }

    @Test
    public void testHW() {
        Solution solution = new Solution();
        int n1 = 128;
        assertEquals(1, solution.hammingWeight(n1));
    }

    @Test
    public void testSN() {
        Solution solution = new Solution();
        int[] n1 = new int[]{4};
        assertEquals(4, solution.singleNumber(n1));
        int[] n2 = new int[]{3, 4, 7, 4, 3};
        assertEquals(7, solution.singleNumber(n2));
    }

    @Test
    public void testSN2() {
        Solution solution = new Solution();
        int[] n1 = new int[]{2, 2, 3, 2};
        assertEquals(3, solution.singleNumber2(n1));
    }

    @Test
    public void testAR() {
        Solution solution = new Solution();
        int l1 = 1;
        int r1 = 2147483647;
        assertEquals(0, solution.rangeBitwiseAnd(l1, r1));
    }

    @Test
    public void testPalinInt() {
        Solution solution = new Solution();
        assertFalse(solution.isPalindrome(-55));
        assertTrue(solution.isPalindrome(1001));
    }

    @Test
    public void testFactZero() {
        Solution solution = new Solution();
        assertEquals(7, solution.trailingZeroes(30));
    }

    @Test
    public void testSQRT() {
        Solution solution = new Solution();
        assertEquals(46340, solution.mySqrt(2147395600));
    }

    @Test
    public void testPOW() {
        Solution solution = new Solution();
        assertEquals(Math.pow(2.0, -2147483647.0), solution.myPow(2.0, -2147483647));
    }

    @Test
    public void testLinePoints() {
        Solution solution = new Solution();
        int[][] p1 = new int[][]{
                {1, 1},
                {3, 2},
                {5, 3},
                {4, 1},
                {2, 3},
                {1, 4}
        };
        assertEquals(4, solution.maxPoints(p1));
    }

    @Test
    public void testStair() {
        Solution solution = new Solution();
        assertEquals(3, solution.climbStairs(3));
    }

    @Test
    public void testRob() {
        Solution solution = new Solution();
        int[] h1 = new int[] {2, 7, 9, 3, 1};
        assertEquals(12, solution.rob(h1));
    }

    @Test
    public void testSeg() {
        Solution solution = new Solution();
        String s1 = "catsandog";
        List<String> d1 = new ArrayList<>();
        d1.add("cats");
        d1.add("dog");
        d1.add("sand");
        d1.add("and");
        d1.add("cat");
        assertFalse(solution.wordBreak(s1, d1));
    }

    @Test
    public void testCoins() {
        Solution solution = new Solution();
        int[] c1= new int[] {1, 2, 5};
        int a1 = 11;
        assertEquals(3, solution.coinChange(c1, a1));
        int[] c2 = new int[] {1};
        int a2 = 0;
        assertEquals(0, solution.coinChange(c2, a2));
    }

    @Test
    public void testLIS() {
        Solution solution = new Solution();
        int[] n1 = new int[] {10, 9, 2, 5, 3, 7, 101, 18};
        assertEquals(4, solution.lengthOfLIS(n1));
    }

    @Test
    public void testTMPS() {
        Solution solution = new Solution();
        List<List<Integer>> lists = new ArrayList<>();
        List<Integer> l10 = new ArrayList<>();
        l10.add(2);
        List<Integer> l11 = new ArrayList<>();
        l11.add(3);
        l11.add(4);
        List<Integer> l12 = new ArrayList<>();
        l12.add(6);
        l12.add(5);
        l12.add(7);
        List<Integer> l13 = new ArrayList<>();
        l13.add(4);
        l13.add(1);
        l13.add(8);
        l13.add(3);
        lists.add(l10);
        lists.add(l11);
        lists.add(l12);
        lists.add(l13);
        assertEquals(11, solution.minimumTotal(lists));
        lists.clear();
        List<Integer> l20 = new ArrayList<>();
        l20.add(-10);
        lists.add(l20);
        assertEquals(-10, solution.minimumTotal(lists));
    }

    @Test
    public void testMPS() {
        Solution solution = new Solution();
        int[][] g1 = new int[][] {
                {1, 3, 1},
                {1, 5, 1},
                {4, 2, 1}
        };
        assertEquals(7, solution.minPathSum(g1));
    }

    @Test
    public void testUPwoO() {
        Solution solution = new Solution();
        int[][] g1 = new int[][] {
                {0, 0, 0},
                {0, 1, 0},
                {0, 0, 0}
        };
        assertEquals(2, solution.uniquePathsWithObstacles(g1));
        int[][] g2 = new int[][] {
                {1, 0}
        };
        assertEquals(0, solution.uniquePathsWithObstacles(g2));
    }

    @Test
    public void testLPL() {
        Solution solution = new Solution();
        String s1 = "aaaa";
        assertEquals("aaaa", solution.longestPalindrome(s1));
    }

    @Test
    public void testInterleave() {
        Solution solution = new Solution();
        String s1 = "aabcc";
        String s2 = "dbbca";
        String s3 = "aadbbcbcac";
        assertTrue(solution.isInterleave(s1, s2, s3));
        String s4 = "aadbbbaccc";
        assertFalse(solution.isInterleave(s1, s2, s4));
    }

    @Test
    public void testED() {
        Solution solution = new Solution();
        String w1 = "intention";
        String w2 = "execution";
        assertEquals(5, solution.minDistance(w1, w2));
    }

    @Test
    public void testStock() {
        Solution solution = new Solution();
        int[] p1 = new int[] {3, 3, 5, 0, 0, 3, 1, 4};
        assertEquals(6, solution.maxProfit3(p1));
    }

    @Test
    public void testStockGeneral() {
        Solution solution = new Solution();
        int[] p1 = new int[] {3, 2, 6, 5, 0, 3};
        assertEquals(7, solution.maxProfit4(2, p1));
    }

    @Test
    public void testMaxSquare() {
        Solution solution = new Solution();
        char[][] m1 = new char[][] {
                {'1', '1', '1', '1', '0'},
                {'1', '1', '1', '1', '0'},
                {'1', '1', '1', '1', '1'},
                {'1', '1', '1', '1', '1'},
                {'0', '0', '1', '1', '1'}
        };
        assertEquals(16, solution.maximalSquare(m1));
        char[][] m2 = new char[][] {
                {'0', '0', '0', '0', '0'},
                {'0', '0', '0', '0', '0'},
                {'0', '0', '0', '0', '1'},
                {'0', '0', '0', '0', '0'}
        };
        assertEquals(1, solution.maximalSquare(m2));
    }
}
