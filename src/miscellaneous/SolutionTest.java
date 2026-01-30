package miscellaneous;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
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

    @Test
    public void testCorridor() {
        Solution solution = new Solution();
        String corridor = "SSPPSPS";
        assertEquals(3, solution.numberOfWays(corridor));
    }

    @Test
    public void testSmooth() {
        Solution solution = new Solution();
        int[] prices = new int[] {3, 2, 1, 4};
        assertEquals(7, solution.getDescentPeriods(prices));
    }

    @Test
    public void testProfit() {
        Solution solution = new Solution();
        int n = 2;
        int[] present = {3, 4};
        int[] future = {5, 8};
        int[][] hierarchy = {{1, 2}};
        int budget = 4;
        assertEquals(4, solution.maxProfit(n, present, future, hierarchy, budget));
    }

    @Test
    public void testStockV() {
        Solution solution = new Solution();
        int[] prices = {1, 7, 9, 8, 2};
        assertEquals(14, solution.maximumProfit(prices, 2));
    }

    @Test
    public void testStockStrat() {
        Solution solution = new Solution();
        int[] prices = new int[] {5, 4, 3};
        int[] strategy = new int[] {1, 1, 0};
        assertEquals(9, solution.maxProfit(prices, strategy, 2));
    }

    @Test
    public void testSecret() {
        Solution solution = new Solution();
        int[][] meetings = new int[][] {{1, 2, 5}, {2, 3, 8}, {1, 5, 10}};
        List<Integer> exp = new ArrayList<>();
        exp.add(0);
        exp.add(1);
        exp.add(2);
        exp.add(3);
        exp.add(5);
        assertEquals(new HashSet<>(exp), new HashSet<>(solution.findAllPeople(6, meetings, 1)));
    }

    @Test
    public void testSortedCol() {
        Solution solution = new Solution();
        String[] strings = new String[] {"cba", "daf", "ghi"};
        assertEquals(1, solution.minDeletionSize(strings));
    }

    @Test
    public void testSortedColII() {
        Solution solution = new Solution();
        String[] strings = new String[] {"xga","xfb","yfa"};
        assertEquals(1, solution.minDeletionSizeII(strings));
    }

    @Test
    public void testTwoEvents() {
        Solution solution = new Solution();
        int[][] events = new int[][] {
                {1, 3, 2}, {4, 5, 2}, {2, 4, 3}
        };
        assertEquals(4, solution.maxTwoEvents(events));
    }

    @Test
    public void testApple() {
        Solution solution = new Solution();
        int[] apples = new int[] {5, 5, 5};
        int[] boxes = new int[] {2, 4, 2, 7};
        assertEquals(4, solution.minimumBoxes(apples, boxes));
    }

    @Test
    public void testHappy() {
        Solution solution = new Solution();
        int[] happiness = new int[] {1, 2, 3};
        int k = 2;
        assertEquals(4, solution.maximumHappinessSum(happiness, k));
    }

    @Test
    public void testPenalty() {
        Solution solution = new Solution();
        String log = "YYNY";
        assertEquals(2, solution.bestClosingTime(log));
    }

    @Test
    public void testMeetRoomIII() {
        Solution solution = new Solution();
        int[][] meetings = new int[][] {
                {0, 10}, {1, 5}, {2, 7}, {3, 4}
        };
        assertEquals(0, solution.mostBooked(2, meetings));
    }

    @Test
    public void testNegaGrid() {
        Solution solution = new Solution();
        int[][] grid = new int[][] {
                {4,3,2,-1}, {3,2,1,-1}, {1,1,-1,-2}, {-1,-1,-2,-3}
        };
        assertEquals(8, solution.countNegatives(grid));
    }

    @Test
    public void testPyramid() {
        Solution solution = new Solution();
        String bottom = "AFFFFA";
        List<String> allowed = new ArrayList<>(List.of("ADA","ADC","ADB","AEA","AEC","AEB","AFA","AFC",
                "AFB","CDA","CDC","CDB","CEA","CEC","CEB","CFA","CFC","CFB","BDA","BDC","BDB","BEA","BEC","BEB",
                "BFA","BFC","BFB","DAA","DAC","DAB","DCA","DCC","DCB","DBA","DBC","DBB","EAA","EAC","EAB","ECA",
                "ECC","ECB","EBA","EBC","EBB","FAA","FAC","FAB","FCA","FCC","FCB","FBA","FBC","FBB","DDA","DDC",
                "DDB","DEA","DEC","DEB","DFA","DFC","DFB","EDA","EDC","EDB","EEA","EEC","EEB","EFA","EFC","EFB",
                "FDA","FDC","FDB","FEA","FEC","FEB","FFA","FFC","FFB","DDD","DDE","DDF","DED","DEE","DEF","DFD",
                "DFE","DFF","EDD","EDE","EDF","EED","EEE","EEF","EFD","EFE","EFF","FDD","FDE","FDF","FED","FEE",
                "FEF","FFD","FFE","FFF"));
        assertFalse(solution.pyramidTransition(bottom, allowed));
    }

    @Test
    public void testMagicSquare() {
        Solution solution = new Solution();
        int[][] grid = new int[][] {
                {4, 3, 8, 4}, {9, 5, 1, 9}, {2, 7, 6, 2}
        };
        assertEquals(1, solution.numMagicSquaresInside(grid));
    }

    @Test
    public void testAddOne() {
        Solution solution = new Solution();
        int[] number = new int[] {9, 9, 9};
        int[] exp = new int[] {1, 0, 0, 0};
        assertArrayEquals(exp, solution.plusOne(number));
    }

    @Test
    public void testRepeat() {
        Solution solution = new Solution();
        int[] nums = new int[] {5,1,5,2,5,3,5,4};
        assertEquals(5, solution.repeatedNTimes(nums));
    }

    @Test
    public void testN3Grid() {
        Solution solution = new Solution();
        assertEquals(30228214, solution.numOfWays(5000));
    }

    @Test
    public void test4Div() {
        Solution solution = new Solution();
        int[] nums = new int[] {100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,100000,
                100000,10000};
        assertEquals(0, solution.sumFourDivisors(nums));
    }

    @Test
    public void testMatrixSumNeg() {
        Solution solution = new Solution();
        int[][] matrix = new int[][] {
                {2, 9, 3}, {5, 4, -4}, {1, 7, 1}
        };
        assertEquals(34, solution.maxMatrixSum(matrix));
    }

    @Test
    public void testMaxLevel() {
        Solution solution = new Solution();
        Solution.TreeNode n1 = new Solution.TreeNode(1);
        Solution.TreeNode n2 = new Solution.TreeNode(7);
        Solution.TreeNode n3 = new Solution.TreeNode(0);
        Solution.TreeNode n4 = new Solution.TreeNode(7);
        Solution.TreeNode n5 = new Solution.TreeNode(-8);
        n1.left = n2;
        n1.right = n3;
        n2.left = n4;
        n2.right = n5;
        assertEquals(2, solution.maxLevelSum(n1));
    }

    @Test
    public void testMaxMulti() {
        Solution solution = new Solution();
        Solution.TreeNode n1 = new Solution.TreeNode(1);
        Solution.TreeNode n2 = new Solution.TreeNode(2);
        Solution.TreeNode n3 = new Solution.TreeNode(3);
        Solution.TreeNode n4 = new Solution.TreeNode(4);
        Solution.TreeNode n5 = new Solution.TreeNode(5);
        Solution.TreeNode n6 = new Solution.TreeNode(6);
        n1.right = n2;
        n2.left = n3;
        n2.right = n4;
        n4.left = n5;
        n4.right = n6;
        assertEquals(90, solution.maxProduct(n1));
    }

    @Test
    public void testMaxDot() {
        Solution solution = new Solution();
        int[] n1 = new int[] {2, 1, -2, 5};
        int[] n2 = new int[] {3, 0, -6};
        assertEquals(18, solution.maxDotProduct(n1, n2));
    }

    @Test
    public void testMinASCII() {
        Solution solution = new Solution();
        String s1 = "delete";
        String s2 = "leet";
        assertEquals(403, solution.minimumDeleteSum(s1, s2));
    }

    @Test
    public void testMatrixRectangle() {
        Solution solution = new Solution();
        char[][] matrix = new char[][] {
                {'1', '0', '1', '0', '0'},
                {'1', '0', '1', '1', '1'},
                {'1', '1', '1', '1', '1'},
                {'1', '0', '0', '1', '0'}
        };
        assertEquals(6, solution.maximalRectangle(matrix));
    }

    @Test
    public void testMinDist() {
        Solution solution = new Solution();
        int[][] points = new int[][] {
                {1, 1}, {3, 4}, {-1, 0}
        };
        assertEquals(7, solution.minTimeToVisitAllPoints(points));
    }

    @Test
    public void testSepSquare() {
        Solution solution = new Solution();
        int[][] squares = new int[][] {
                {0, 0, 2}, {1, 1, 1}
        };
        assertTrue(Math.abs(1.16667-solution.separateSquares(squares))<=0.00001);
    }

    @Test
    public void testSeqSquareII() {
        Solution solution = new Solution();
        int[][] squares = new int[][] {
                {0, 0, 2}, {1, 1, 1}
        };
        assertTrue(Math.abs(1.00000-solution.separateSquaresII(squares))<=0.00001);
    }

    @Test
    public void testMaxSquare() {
        Solution solution = new Solution();
        int[] hb = new int[] {3, 2, 4};
        int[] vb = new int[] {4, 6, 7, 12, 10, 13, 2};
        assertEquals(9, solution.maximizeSquareHoleArea(3, 13, hb, vb));
    }

    @Test
    public void testMaxFenceSquare() {
        Solution solution = new Solution();
        int[] hf = new int[] {2, 3};
        int[] vf = new int[] {2};
        assertEquals(4, solution.maximizeSquareArea(4, 3, hf, vf));
    }

    @Test
    public void testIntersectSquare() {
        Solution solution = new Solution();
        int[][] bl = new int[][] {
                {1, 1}, {1, 3}, {1, 5}
        };
        int[][] tr = new int[][] {
                {5, 5}, {5, 7}, {5, 9}
        };
        assertEquals(4, solution.largestSquareArea(bl, tr));
    }

    @Test
    public void testMaxMagicSquare() {
        Solution solution = new Solution();
        int[][] grid = new int[][] {
                {7,1,4,5,6},
                {2,5,1,6,4},
                {1,5,4,3,2},
                {1,2,7,3,4}
        };
        assertEquals(3, solution.largestMagicSquare(grid));
    }

    @Test
    public void testMaxSumSquare() {
        Solution solution = new Solution();
        int[][] matrix = new int[][] {
                {1,1,3,2,4,3,2},
                {1,1,3,2,4,3,2},
                {1,1,3,2,4,3,2}
        };
        assertEquals(2, solution.maxSideLength(matrix, 4));
    }

    @Test
    public void testMinBit() {
        Solution solution = new Solution();
        List<Integer> nums = new ArrayList<>();
        nums.add(2);
        nums.add(3);
        nums.add(5);
        nums.add(7);
        int[] exp = new int[] {-1, 1, 4, 3};
        assertArrayEquals(exp, solution.minBitwiseArray(nums));
    }

    @Test
    public void testMinPairRemove() {
        Solution solution = new Solution();
        int[] nums = new int[] {5, 2, 3, 1};
        assertEquals(2, solution.minimumPairRemoval(nums));
    }

    @Test
    public void testMinPairRemoveII() {
        Solution solution = new Solution();
        int[] nums = new int[] {3,4,1,1,-3,2,4,3};
        assertEquals(5, solution.minimumPairRemovalII(nums));
    }

    @Test
    public void testMinMaxPair() {
        Solution solution = new Solution();
        int[] nums = new int[] {3,5,4,2,4,6};
        assertEquals(8, solution.minPairSum(nums));
    }

    @Test
    public void testMinMaxDiff() {
        Solution solution = new Solution();
        int[] nums = new int[] {9, 4, 1, 7};
        assertEquals(2, solution.minimumDifference(nums, 2));
    }

    @Test
    public void testMinDiffPairs() {
        int[] arr = {4, 2, 1, 3};
        Solution solution = new Solution();
        List<List<Integer>> actual = solution.minimumAbsDifference(arr);
        List<List<Integer>> expected = List.of(
                List.of(1, 2),
                List.of(2, 3),
                List.of(3, 4)
        );
        assertEquals(expected, actual);
    }

    @Test
    public void testSwitchMinCost() {
        Solution solution = new Solution();
        int[][] edges = new int[][] {
                {0, 1, 3}, {3, 1, 1}, {2, 3, 4}, {0, 2, 2}
        };
        assertEquals(5, solution.minCost(4, edges));
    }

    @Test
    public void testTeleMinCost() {
        Solution solution = new Solution();
        int[][] grid = new int[][] {
                {1, 3, 3}, {2, 5, 4}, {4, 3, 5}
        };
        assertEquals(7, solution.minCost(grid, 2));
    }

    @Test
    public void testMinCostStringTransform() {
        Solution solution = new Solution();
        String source = "abcd";
        String target = "acbe";
        char[] original = new char[] {'a', 'b', 'c', 'c', 'e', 'd'};
        char[] changed = new char[] {'b', 'c', 'b', 'e', 'b', 'e'};
        int[] cost = new int[] {2, 5, 5, 1, 2, 20};
        assertEquals(28, solution.minimumCost(source, target, original, changed, cost));
    }

    @Test
    public void testMinCostStringTransformII() {
        Solution solution = new Solution();
        String source = "abcd";
        String target = "acbe";
        String[] original = new String[] {"a","b","c","c","e","d"};
        String[] changed = new String[] {"b","c","b","e","b","e"};
        int[] cost = new int[] {2,5,5,1,2,20};
        assertEquals(28, solution.minimumCost(source, target, original, changed, cost));
    }
}
