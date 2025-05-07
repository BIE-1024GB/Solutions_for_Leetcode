package leetcode_75;

import java.util.*;

/**
 * @author Jiarui BIE
 * @version 1.0
 * @since 2025/04/24
 */
public class Solution {
    public String mergeAlternately(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        int l = word1.length();
        int r = word2.length();
        int limit = Math.min(l, r);
        for (int i = 0; i <= limit-1; i++) {
            sb.append(word1.charAt(i));
            sb.append(word2.charAt(i));
        }
        if (limit < l) {
            for (int i = limit; i <= l-1; i++) sb.append(word1.charAt(i));
        }
        if (limit < r) {
            for (int i = limit; i <= r-1; i++) sb.append(word2.charAt(i));
        }
        return sb.toString();
    }

    private boolean isDivisor(String div, String str) {
        if (str.length()%div.length() != 0) {
            return false;
        } else {
            if (str.length() == div.length()) {
                return str.equals(div);
            } else {
                return (div.equals(str.substring(0, div.length()))) && isDivisor(div, str.substring(div.length()));
            }
        }
    }
    public String gcdOfStrings(String str1, String str2) {
        if (str1.length() > str2.length()) {
            return gcdOfStrings(str2, str1);
        } else {
            String gcd = "";
            for (int i = 1; i <= str1.length(); i++) {
                String d = str1.substring(0, i);
                if (isDivisor(d, str1) && isDivisor(d, str2)) gcd = d;
            }
            return gcd;
        }
    }

    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int mc = Integer.MIN_VALUE;
        for (int c: candies) {
            if (c > mc) {
                mc = c;
            }
        }
        List<Boolean> extra = new ArrayList<>();
        for (int i = 0; i <= candies.length-1; i++) {
            if (candies[i]+extraCandies >= mc) {
                extra.add(true);
            } else {
                extra.add(false);
            }
        }
        return extra;
    }

    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        if (n == 0) return true;
        if (flowerbed.length == 1) {
            if (flowerbed[0] == 0) {
                return 1 >= n;
            } else {
                return 0 >= n;
            }
        }
        int ap = 0;
        for (int i = 0; i <= flowerbed.length-1; i++) {
            if (flowerbed[i] == 0) {
                if (i == 0) {
                    if (flowerbed[1] == 0) {
                        ap += 1;
                        flowerbed[i] = 1;
                    }
                } else if (i == flowerbed.length-1) {
                    if (flowerbed[flowerbed.length-2] == 0) {
                        ap += 1;
                        flowerbed[i] = 1;
                    }
                } else {
                    if (flowerbed[i-1] == 0 && flowerbed[i+1] == 0) {
                        ap += 1;
                        flowerbed[i] = 1;
                    }
                }
            }
            if (ap >= n) return true;
        }
        return false;
    }

    public String reverseVowels(String s) {
        if (s.length() == 1) {
            return s;
        }
        String vowels = "AEIOUaeiou";
        int lp = 0;
        int rp = s.length()-1;
        char[] sc = s.toCharArray();
        while (lp < rp) {
            if (vowels.indexOf(s.charAt(lp)) == -1) {
                lp++;
            } else if (vowels.indexOf(s.charAt(rp)) == -1) {
                rp--;
            } else {
                char temp = sc[lp];
                sc[lp] = sc[rp];
                sc[rp] = temp;
                lp++;
                rp--;
            }
        }
        return new String(sc);
    }

    public String reverseWords(String s) {
        String[] words = s.trim().split("\\s+");
        String res = words[words.length-1];
        for (int i = words.length-2; i >= 0; i--) res = res.concat(" ").concat(words[i]);
        return res;
    }

    public int[] productExceptSelf(int[] nums) {
        int[] answer = new int[nums.length];
        int[] lv = new int[nums.length];
        int[] rv = new int[nums.length];
        for (int i = 0; i <= lv.length-1; i++) {
            if (i == 0) {
                lv[i] = 1;
            } else {
                lv[i] = nums[i-1]*lv[i-1];
            }
        }
        for (int i = rv.length-1; i >= 0; i--) {
            if (i == rv.length-1) {
                rv[i] = 1;
            } else {
                rv[i] = nums[i+1]*rv[i+1];
            }
        }
        for (int i = 0; i <= answer.length-1; i++) {
            answer[i] = lv[i]*rv[i];
        }
        return answer;
    }

    public boolean increasingTriplet(int[] nums) {
        if (nums.length < 3) return false;
        int fv = Integer.MAX_VALUE;
        int sv = Integer.MAX_VALUE;
        for (int n: nums) {
            if (n > sv) {
                return true;
            } else if (n > fv && n < sv) {
                sv = n;
            } else if (n < fv) {
                fv = n;
            }
        }
        return false;
    }
}
