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
        int[][] heights = new int[][] {
                {1,4,3,1,3,2},
                {3,2,1,3,2,4},
                {2,3,3,2,3,1}
        };
        assertEquals(4, solution.trapRainWater(heights));
    }

    @Test
    public void testCW() {
        Solution solution = new Solution();
        int[] heights = new int[] {1,8,6,2,5,4,8,3,7};
        assertEquals(49, solution.maxArea(heights));
    }

    @Test
    public void testMSS() {
        Solution solution = new Solution();
        int[] nums = new int[] {1, 2, 3};
        assertEquals(2, solution.subarraySum(nums, 3));
    }

    @Test
    public void testBrew() {
        Solution solution = new Solution();
        int[] skills = new int[] {1, 5, 2, 4};
        int[] manas = new int[] {5, 1, 4, 2};
        assertEquals(110, solution.minTime(skills, manas));
    }

    @Test
    public void testEnergy() {
        Solution solution = new Solution();
        int[] energy = new int[] {5,2,-10,-5,1};
        assertEquals(3, solution.maximumEnergy(energy, 3));
    }

    @Test
    public void testPower() {
        Solution solution = new Solution();
        int[] powers = new int[] {7,1,6,6};
        assertEquals(13, solution.maximumTotalDamage(powers));
    }

    @Test
    public void testRemoveAna() {
        Solution solution = new Solution();
        String[] words = new String[] {"nelduncd","dcnndeul","uendlcnd","nluncedd","fozlsvr","osfvrlz","vozsrfl","dm",
                "md","md","dm","md","dm","md","md","dm","dm","dm","dm","md","eatzkewuyx","a","wulzacir"};
        List<String> exp = new ArrayList<>();
        exp.add("nelduncd");
        exp.add("fozlsvr");
        exp.add("dm");
        exp.add("eatzkewuyx");
        exp.add("a");
        exp.add("wulzacir");
        List<String> act = solution.removeAnagrams(words);
        assertEquals(exp.size(), act.size());
        for (int i = 0; i <= exp.size()-1; i++) {
            assertEquals(exp.get(i), act.get(i));
        }
    }
}
