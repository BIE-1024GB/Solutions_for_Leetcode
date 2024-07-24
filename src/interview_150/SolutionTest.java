package interview_150;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SolutionTest {
    @Test
    public void testSingleElementArray() {
        int[] nums = {1};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(1, result);
    }

    @Test
    public void testTwoElementArray() {
        int[] nums = {1, 2};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(2, result);
    }

    @Test
    public void testNoDuplicates() {
        int[] nums = {1, 2, 3, 4, 5};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(5, result);
    }

    @Test
    public void testAllDuplicates() {
        int[] nums = {1, 1, 1, 1, 1};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(2, result); // Only one duplicate allowed
    }

    @Test
    public void testMixedDuplicates() {
        int[] nums = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(10, result); // Each number appears twice
    }

    @Test
    public void testMoreThanTwoDuplicates() {
        int[] nums = {1, 1, 1, 2, 2, 2, 3, 3, 3};
        int result = new Solution().removeDuplicates(nums);
        assertEquals(6, result); // Only two of each number allowed
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
}