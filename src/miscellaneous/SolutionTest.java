package miscellaneous;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class SolutionTest {
    @Test
    public void testLang() {
        Solution solution = new Solution();
        int n = 3;
        int[][] languages = new int[][]{
                {2},
                {1, 3},
                {1, 2},
                {3}
        };
        int[][] fri = new int[][]{
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
        int[] sides = new int[]{1, 2, 1, 10};
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
        int[] vertex = new int[]{1, 3, 1, 4, 1, 5};
        assertEquals(13, solution.minScoreTriangulation(vertex));
    }

    @Test
    public void testTS() {
        Solution solution = new Solution();
        int[] arr = new int[]{1, 2, 3, 4, 5};
        assertEquals(8, solution.triangularSum(arr));
    }

    @Test
    public void testBottle() {
        Solution solution = new Solution();
        assertEquals(19, solution.numWaterBottles(15, 4));
    }

    @Test
    public void testBottle2() {
        Solution solution = new Solution();
        assertEquals(15, solution.maxBottlesDrunk(13, 6));
    }

    @Test
    public void testRain2() {
        Solution solution = new Solution();
        int[][] heights = new int[][]{
                {1, 4, 3, 1, 3, 2},
                {3, 2, 1, 3, 2, 4},
                {2, 3, 3, 2, 3, 1}
        };
        assertEquals(4, solution.trapRainWater(heights));
    }

    @Test
    public void testCW() {
        Solution solution = new Solution();
        int[] heights = new int[]{1, 8, 6, 2, 5, 4, 8, 3, 7};
        assertEquals(49, solution.maxArea(heights));
    }

    @Test
    public void testMSS() {
        Solution solution = new Solution();
        int[] nums = new int[]{1, 2, 3};
        assertEquals(2, solution.subarraySum(nums, 3));
    }

    @Test
    public void testBrew() {
        Solution solution = new Solution();
        int[] skills = new int[]{1, 5, 2, 4};
        int[] manas = new int[]{5, 1, 4, 2};
        assertEquals(110, solution.minTime(skills, manas));
    }

    @Test
    public void testEnergy() {
        Solution solution = new Solution();
        int[] energy = new int[]{5, 2, -10, -5, 1};
        assertEquals(3, solution.maximumEnergy(energy, 3));
    }

    @Test
    public void testPower() {
        Solution solution = new Solution();
        int[] powers = new int[]{7, 1, 6, 6};
        assertEquals(13, solution.maximumTotalDamage(powers));
    }

    @Test
    public void testRemoveAna() {
        Solution solution = new Solution();
        String[] words = new String[]{"nelduncd", "dcnndeul", "uendlcnd", "nluncedd", "fozlsvr", "osfvrlz", "vozsrfl", "dm",
                "md", "md", "dm", "md", "dm", "md", "md", "dm", "dm", "dm", "dm", "md", "eatzkewuyx", "a", "wulzacir"};
        List<String> exp = new ArrayList<>();
        exp.add("nelduncd");
        exp.add("fozlsvr");
        exp.add("dm");
        exp.add("eatzkewuyx");
        exp.add("a");
        exp.add("wulzacir");
        List<String> act = solution.removeAnagrams(words);
        assertEquals(exp.size(), act.size());
        for (int i = 0; i <= exp.size() - 1; i++) {
            assertEquals(exp.get(i), act.get(i));
        }
    }

    @Test
    public void testAI() {
        Solution solution = new Solution();
        List<Integer> nums = new ArrayList<>();
        nums.add(5);
        nums.add(8);
        nums.add(-2);
        nums.add(-1);
        assertTrue(solution.hasIncreasingSubarrays(nums, 2));
    }

    @Test
    public void testkAI() {
        Solution solution = new Solution();
        List<Integer> nums = new ArrayList<>(Arrays.asList(1,2,3,4,4,4,4,5,6,7));
        assertEquals(2, solution.maxIncreasingSubarrays(nums));
    }

    @Test
    public void testLaser() {
        Solution solution = new Solution();
        String[] devices = new String[] {"011001","000000","010100","001000"};
        assertEquals(8, solution.numberOfBeams(devices));
    }

    @Test
    public void testDir() {
        Solution solution = new Solution();
        int[] nums = new int[] {16,13,10,0,0,0,10,6,7,8,7};
        assertEquals(3, solution.countValidSelections(nums));
    }

    @Test
    public void testBit() {
        Solution solution = new Solution();
        assertEquals(15, solution.smallestNumber(10));
    }

    @Test
    public void testOp() {
        Solution solution = new Solution();
        int[] target = new int[] {3, 1, 1, 2};
        assertEquals(4, solution.minNumberOperations(target));
    }

    @Test
    public void testBalloon() {
        Solution solution = new Solution();
        String colors = "abaac";
        int[] nt = new int[] {1, 2, 3, 4, 5};
        assertEquals(3, solution.minCost(colors, nt));
    }

    @Test
    public void testMaxMinPower() {
        Solution solution = new Solution();
        int[] stations = new int[] {1, 2, 4, 5, 0};
        assertEquals(5, solution.maxPower(stations, 1, 2));
    }

    @Test
    public void testTurnZero() {
        Solution solution = new Solution();
        assertEquals(4, solution.minimumOneBitOperations(6));
    }

    @Test
    public void testZeroOps() {
        Solution solution = new Solution();
        int[] nums = new int[]{3, 1, 2, 1};
        assertEquals(3, solution.minOperations(nums));
    }

    @Test
    public void testZO() {
        Solution solution = new Solution();
        String[] strs = new String[] {"00011", "00001", "00001", "0011", "111"};
        assertEquals(3, solution.findMaxForm(strs, 8, 5));
    }

    @Test
    public void testTurnOne() {
        Solution solution = new Solution();
        int[] nums = new int[] {2, 6, 3, 4};
        assertEquals(4, solution.minOperationsOne(nums));
    }

    @Test
    public void testDominantOne() {
        Solution solution = new Solution();
        String string = "00011";
        assertEquals(5, solution.numberOfSubstrings(string));
    }

    @Test
    public void testKApart() {
        Solution solution = new Solution();
        int[] nums = new int[] {1,0,0,0,1,0,0,1};
        assertTrue(solution.kLengthApart(nums, 2));
    }

    @Test
    public void testBits() {
        Solution solution = new Solution();
        int[] bits = new int[] {1, 1, 1, 0};
        assertFalse(solution.isOneBitCharacter(bits));
    }

    @Test
    public void testDouble() {
        Solution solution = new Solution();
        int[] nums = new int[] {5, 3, 6, 1, 12};
        assertEquals(24, solution.findFinalValue(nums, 3));
    }

    @Test
    public void testDivide3() {
        Solution solution = new Solution();
        int[] nums = new int[] {1, 2, 3, 4};
        assertEquals(3, solution.minimumOperations(nums));
    }

    @Test
    public void testMaxDivide3() {
        Solution solution = new Solution();
        int[] nums = new int[] {3, 6, 5, 1, 8};
        assertEquals(18, solution.maxSumDivThree(nums));
    }

    @Test
    public void testDivideK() {
        Solution solution = new Solution();
        assertEquals(3, solution.smallestRepunitDivByK(3));
    }

    @Test
    public void testkMSS() {
        Solution solution = new Solution();
        int[] nums = new int[] {-5,1,2,-3,4};
        assertEquals(4, solution.maxSubarraySum(nums, 2));
    }

    @Test
    public void testMinOp() {
        Solution solution = new Solution();
        int[] nums = new int[] {3, 2};
        assertEquals(5, solution.minOperations(nums, 6));
    }

    @Test
    public void testRMS() {
        Solution solution = new Solution();
        int[] nums = new int[] {3, 1, 4, 2};
        assertEquals(1, solution.minSubarray(nums, 6));
    }

    @Test
    public void test4Sum() {
        Solution solution = new Solution();
        int[] nums = new int[] {1000000000,1000000000,1000000000,1000000000};
        int target = -294967296;
        assertEquals(0, solution.fourSum(nums, target).size());
    }

    @Test
    public void testTrape() {
        Solution solution = new Solution();
        int[][] points = new int[][] {
                {-3, -70}, {8, -70}, {-85, 90},
                {-99, 90}, {-6, 90}, {47, -23},
                {-16, -23}
        };
        assertEquals(7, solution.countTrapezoids(points));
    }

    @Test
    public void testCrash() {
        Solution solution = new Solution();
        String pos = "RLRSLL";
        assertEquals(5, solution.countCollisions(pos));
    }

    @Test
    public void testPartition() {
        Solution solution = new Solution();
        int[] nums = new int[] {10, 10, 3, 7, 6};
        assertEquals(4, solution.countPartitions(nums));
    }

    @Test
    public void testPartitionII() {
        Solution solution = new Solution();
        int[] nums = new int[] {9,4,1,3,7};
        assertEquals(6, solution.countPartitions(nums, 4));
    }

    @Test
    public void testOdd() {
        Solution solution = new Solution();
        assertEquals(3, solution.countOdds(3, 7));
    }

    @Test
    public void testSquare() {
        Solution solution = new Solution();
        assertEquals(4, solution.countTriples(10));
    }

    @Test
    public void testTriplets() {
        Solution solution = new Solution();
        int[] nums = new int[] {8, 4, 2, 8, 4};
        assertEquals(2, solution.specialTriplets(nums));
    }

    @Test
    public void testUnlock() {
        Solution solution = new Solution();
        int[] comp = new int[] {38,223,100,123,406,234,256,93,222,259,233,69,139,245,45,98,214};
        assertEquals(789741546, solution.countPermutations(comp));
    }

    @Test
    public void testBuilding() {
        Solution solution = new Solution();
        int[][] buildings = new int[][] {
                {1, 2}, {2, 2}, {3, 2}, {2, 1}, {2, 3}
        };
        assertEquals(1, solution.countCoveredBuildings(3, buildings));
    }

    @Test
    public void testMention() {
        Solution solution = new Solution();
        List<String> l1 = new ArrayList<>();
        l1.add("MESSAGE");
        l1.add("1");
        l1.add("id0 id1");
        List<String> l2 = new ArrayList<>();
        l2.add("MESSAGE");
        l2.add("5");
        l2.add("id2");
        List<String> l3 = new ArrayList<>();
        l3.add("MESSAGE");
        l3.add("6");
        l3.add("ALL");
        List<String> l4 = new ArrayList<>();
        l4.add("OFFLINE");
        l4.add("5");
        l4.add("2");
        List<List<String>> events = new ArrayList<>();
        events.add(l1);
        events.add(l2);
        events.add(l3);
        events.add(l4);
        int[] exp = new int[] {2, 2, 2};
        int[] act = solution.countMentions(3, events);
        for (int i = 0; i <= exp.length-1; i++) {
            assertEquals(exp[i], act[i]);
        }
    }
}
