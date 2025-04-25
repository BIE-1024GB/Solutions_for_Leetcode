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
}
