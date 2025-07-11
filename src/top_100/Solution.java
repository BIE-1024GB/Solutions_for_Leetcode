package top_100;

import java.util.*;

/**
 * @author Jiarui BIE
 * @version 1.0
 * @since 2025/07/08
 */
public class Solution {
    private void LCdfs(List<String> res, StringBuilder sb, String digits) {
        if (sb.length() == digits.length()) {
            res.add(sb.toString());
        } else {
            char[] letters;
            switch (digits.charAt(sb.length())) {
                case '2' -> letters = new char[] {'a', 'b', 'c'};
                case '3' -> letters = new char[] {'d', 'e', 'f'};
                case '4' -> letters = new char[] {'g', 'h', 'i'};
                case '5' -> letters = new char[] {'j', 'k', 'l'};
                case '6' -> letters = new char[] {'m', 'n', 'o'};
                case '7' -> letters = new char[] {'p', 'q', 'r', 's'};
                case '8' -> letters = new char[] {'t', 'u', 'v'};
                default -> letters = new char[] {'w', 'x', 'y', 'z'};
            }
            for (char c : letters) {
                sb.append(c);
                LCdfs(res, sb, digits);
                sb.deleteCharAt(sb.length()-1);
            }
        }
    }
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (!digits.isEmpty()) {
            LCdfs(res, new StringBuilder(), digits);
        }
        return res;
    }

    private void GPdfs(List<String> res, StringBuilder sb, int lp, int rp, int n) {
        if (sb.length() == n*2) {
            res.add(sb.toString());
        } else {
            if (lp < n) {
                sb.append('(');
                GPdfs(res, sb, lp+1, rp, n);
                sb.deleteCharAt(sb.length()-1);
            }
            if (rp < lp) {
                sb.append(')');
                GPdfs(res, sb, lp, rp+1, n);
                sb.deleteCharAt(sb.length()-1);
            }
        }
    }
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        GPdfs(res, new StringBuilder(), 0, 0, n);
        return res;
    }

    private void CSdfs(int[] candidates, int target, List<List<Integer>> res, List<Integer> curr, int sum, int index) {
        if (sum == target) {
            List<Integer> list = new ArrayList<>(curr);    //don't directly add 'curr'
            res.add(list);
        } else if (sum < target) {
            for (int i = index; i <= candidates.length-1; i++) {
                if (sum+candidates[i] > target) {
                    break;
                } else {
                    curr.add(candidates[i]);
                    CSdfs(candidates, target, res, curr, sum+candidates[i], i);
                    curr.removeLast();
                }
            }
        }
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(candidates);
        CSdfs(candidates, target, res, new ArrayList<>(), 0, 0);
        return res;
    }

    private void Pdfs(int[] nums, List<List<Integer>> res, List<Integer> curr) {
        if (curr.size() == nums.length) {
            List<Integer> list = new ArrayList<>(curr);
            res.add(list);
        } else {
            for (int n : nums) {
                if (!curr.contains(n)) {
                    curr.add(n);
                    Pdfs(nums, res, curr);
                    curr.removeLast();
                }
            }
        }
    }
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Pdfs(nums, res, new ArrayList<>());
        return res;
    }

    private boolean isValid(char[][] board, int row, int col, int n) {
        for (int i = 0; i < row; i++) {
            if (board[i][col] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
            if (board[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }
    private List<String> constructSolution(char[][] board) {
        List<String> solution = new ArrayList<>();
        for (char[] row : board) {
            solution.add(new String(row));
        }
        return solution;
    }
    private void backtrack(List<List<String>> solutions, char[][] board, int row, int n) {
        if (row == n) {
            solutions.add(constructSolution(board));
        } else {
            for (int col = 0; col < n; col++) {
                if (isValid(board, row, col, n)) {
                    board[row][col] = 'Q';
                    backtrack(solutions, board, row + 1, n);
                    board[row][col] = '.';
                }
            }
        }
    }
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> solutions = new ArrayList<>();
        char[][] board = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] = '.';
            }
        }
        backtrack(solutions, board, 0, n);
        return solutions;
    }

    private void Sdfs(List<List<Integer>> res, int[] nums, List<Integer> curr, int index) {
        if (curr.size() == nums.length) {
            List<Integer> list = new ArrayList<>(curr);
            res.add(list);
        } else {
            curr.add(nums[index]);
            List<Integer> list = new ArrayList<>(curr);
            res.add(list);
            int k = 1;
            while (index+k <= nums.length-1) {
                Sdfs(res, nums, curr, index+k);
                k++;
            }
            curr.removeLast();
            if (curr.isEmpty() && index <= nums.length-2) {
                Sdfs(res, nums, curr, index+1);
            }
        }
    }
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        res.add(new ArrayList<>(0));
        Sdfs(res, nums, new ArrayList<>(), 0);
        return res;
    }

    private boolean SWdfs(char[][] board, String word, int index, int r, int c, boolean[][] visit) {
        if (index == word.length()) {
            return true;
        } else {
            if (r-1>=0 && !visit[r-1][c] && board[r-1][c]==word.charAt(index)) {
                visit[r-1][c] = true;
                if (SWdfs(board, word, index+1, r-1, c, visit)) {
                    return true;
                }
                visit[r-1][c] = false;
            }
            if (c+1<=board[0].length-1 && !visit[r][c+1] && board[r][c+1]==word.charAt(index)) {
                visit[r][c+1] = true;
                if (SWdfs(board, word, index+1, r, c+1, visit)) {
                    return true;
                }
                visit[r][c+1] = false;
            }
            if (r+1<=board.length-1 && !visit[r+1][c] && board[r+1][c]==word.charAt(index)) {
                visit[r+1][c] = true;
                if (SWdfs(board, word, index+1, r+1, c, visit)) {
                    return true;
                }
                visit[r+1][c] = false;
            }
            if (c-1>=0 && !visit[r][c-1] && board[r][c-1]==word.charAt(index)) {
                visit[r][c-1] = true;
                if (SWdfs(board, word, index+1, r, c-1, visit)) {
                    return true;
                }
                visit[r][c-1] = false;
            }
            return false;
        }
    }
    public boolean exist(char[][] board, String word) {
        boolean[][] visit = new boolean[board.length][board[0].length];
        for (int i = 0; i <= board.length-1; i++) {
            for (int j = 0; j <= board[0].length-1; j++) {
                if (board[i][j] == word.charAt(0)) {
                    visit[i][j] = true;
                    if (SWdfs(board, word, 1, i, j, visit)) {
                        return true;
                    }
                    visit[i][j] = false;
                }
            }
        }
        return false;
    }

    private boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
    private void backtrack(String s, int start, List<String> current, List<List<String>> result) {
        // If we've reached the end of the string, add the current partition to result
        if (start == s.length()) {
            result.add(new ArrayList<>(current));
            return;
        }
        // Explore all possible partitions
        for (int end = start + 1; end <= s.length(); end++) {
            String substring = s.substring(start, end);
            if (isPalindrome(substring)) {
                current.add(substring);
                backtrack(s, end, current, result);
                current.removeLast();
            }
        }
    }
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        backtrack(s, 0, new ArrayList<>(), result);
        return result;
    }
}
