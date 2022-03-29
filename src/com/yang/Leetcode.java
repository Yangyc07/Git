package com.yang;

import com.sun.xml.internal.ws.encoding.MtomCodec;
import sun.security.krb5.internal.PAData;

import java.util.*;
import java.util.concurrent.locks.AbstractQueuedSynchronizer;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;


class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int val) {
        this.val = val;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}


public class Leetcode {
    static List<List<Integer>> nums = new ArrayList<>();
    public static void main(String[] args) {
        int sum = 0;

        List<Integer> path = new ArrayList<>();
        backTrace1(path, 3, 7, 1, 0);
    }

    public long getDescentPeriods(int[] prices) {
        int n = prices.length;
        long[][] f = new long[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j)
                    f[i][j] = 1;
                else {
                    if (j + 1 < n && prices[j] - prices[j + 1] == 1)
                        f[i][j] = f[i][j - 1] + 1;
                    else
                        f[i][j] = f[i][j - 1];
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(f[i][j] + " ");
            }
            System.out.println();
        }
        return f[0][n - 1];
    }

    public String addSpaces(String s, int[] spaces) {
        StringBuilder sb = new StringBuilder();
        int index = 0, n = s.length();
        for (int i = 0; i < spaces.length; i++) {
            int end = spaces[i];
            String str = s.substring(index, end);
            index = end;
            //System.out.println(str);
            if (str != null)
                sb.append(str);
            sb.append(" ");
        }
        String str = s.substring(index, n);
        if (str != null)
            sb.append(str);
        return sb.toString();
    }

    public String firstPalindrome(String[] words) {
        for (String str : words) {
            if (isHui(str))
                return str;
        }
        return "";
    }

    public boolean isHui(String str) {
        int n = str.length();
        for (int i = 0; i < n / 2; i++) {
            if (str.charAt(i) != str.charAt(n - 1 - i))
                return false;
        }
        return true;
    }

    public void nextPermutation(int[] nums) {
        int n = nums.length;
        for (int i = n - 2; i >= 0; i--) {
            //找逆序对
            if (nums[i] < nums[i + 1]) {
                for (int j = i + 1; j < n; j++) { //第一个大于nums[i]的数
                    Arrays.sort(nums, i + 1, n);
                    if (nums[j] > nums[i]) {
                        int temp = nums[i];
                        nums[i] = nums[j];
                        nums[j] = temp;
                        return;
                    }
                }
            }
        }
        Arrays.sort(nums);
    }

    //5952
    public int countPoints(String rings) {
        int count = 0;
        int n = rings.length();
        ArrayList<TreeSet<Character>> value = new ArrayList<>();
        for (int i = 0; i < 10; i++)
            value.add(new TreeSet<Character>());
        for (int i = 0; i < n; i += 2)
            value.get(rings.charAt(i + 1) - '0').add(rings.charAt(i));
        for (int i = 0; i < 10; i++) {
            if (value.get(i).size() == 3)
                count++;
        }
        return count;
    }

    //34
    public int[] searchRange(int[] nums, int target) {
        int[] f = {-1, -1};
        int n = nums.length;
        int l = binarySearch(nums, target, true);
        int r = binarySearch(nums, target, false) - 1;
        if (l <= r && r < n && nums[l] == target && nums[r] == target)
            return new int[]{l, r};
        return f;
    }

    public int binarySearch(int[] nums, int target, boolean flag) {
        int left = 0, right = nums.length - 1, ans = nums.length;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target || (flag && nums[mid] >= target)) {
                right = mid - 1;
                ans = mid;
            } else
                left = mid + 1;
        }
        return ans;
    }

    //238
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int left = 1, right = 1;
        int[] f = new int[n];
        for (int i = 0; i < n; i++) {
            f[i] = left;
            left *= nums[i];
        }
        for (int i = n - 1; i >= 0; i--) {
            f[i] *= right;
            right *= nums[i];
        }
        return f;
    }

    int n = 0, ans = 0;

    public int kthSmallest(TreeNode root, int k) {
        inOrder(root, k);
        return ans;
    }

    void inOrder(TreeNode root, int k) {
        if (root != null) {
            inOrder(root.left, k);
            if (n++ == k) {
                ans = root.val;
                return;
            }
            inOrder(root.right, k);
        }
    }

    ArrayList<Integer> values = new ArrayList<>();
    ArrayList<Integer> value = new ArrayList<>();

    public int[] findEvenNumbers(int[] digits) {
        traceBack(digits, 0, 3, new int[digits.length]);
        Collections.sort(values);
        int ans[] = new int[values.size()];
        int i = 0;
        for (Integer num : values)
            ans[i++] = num;
        return ans;
    }

    public void traceBack(int nums[], int start, int n, int used[]) {
        if (value.size() == n) {
            int num = value.get(0) * 100 + value.get(1) * 10 + value.get(2);
            if (num >= 100 && num % 2 == 0)
                return;
        }

        for (int i = start; i < nums.length; i++) {
            if (used[i] == 1)
                continue;
            if (value.size() == 0 && nums[i] == 0)
                continue;
            if (value.size() == 2 && nums[i] % 2 != 0)
                continue;

            value.add(nums[i]);
            used[i] = 1;
            traceBack(nums, i + 1, n, used);
            used[i] = 0;
        }
    }


    public ListNode deleteMiddle(ListNode head) {
        if (head.next == null)
            return null;
        if (head.next.next == null)
            return head.next;
        ListNode pre = head;
        ListNode p = head.next.next;
        while (pre != null && p.next != null) {
            pre = pre.next;
            if (p.next.next != null)
                p = p.next.next;
        }
        pre.next = pre.next.next;
        return head;

    }

    //383
    public boolean canConstruct(String ransomNote, String magazine) {
        char f[] = new char[26];
        for (int i = 0; i < magazine.length(); i++)
            f[magazine.charAt(i) - 'a']++;
        for (int i = 0; i < ransomNote.length(); i++) {
            if (f[ransomNote.charAt(i) - 'a']-- <= 0)
                return false;
        }
        return true;
    }

    //506
    public String[] findRelativeRanks(int[] score) {
        String[] ss = new String[]{"Gold Medal", "Silver Medal", "Bronze Medal"};
        HashMap<Integer, Integer> map = new HashMap<>();
        int n = score.length;
        String[] f = new String[n];
        for (int i = 0; i < n; i++)
            map.put(score[i], i);
        Arrays.sort(score);
        for (int i = 0; i < n; i++) {
            int j = n - i - 1;
            f[map.get(score[i])] = j <= 2 ? ss[j] : String.valueOf(i + 1);
        }
        return f;
    }

    public List<Integer> grayCode(int n) {
        System.out.println(1 << 3);
        return null;
    }

    //148
    public ListNode sortList(ListNode head) {
        if (head == null)
            return head;
        ListNode mid = findMiddle(head);
        ListNode right = mid.next;
        mid.next = null;

        sortList(head);
        sortList(right);

        return merge(head, right);
    }

    public ListNode findMiddle(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    public ListNode merge(ListNode l1, ListNode l2) {
        ListNode l = new ListNode(-1);
        ListNode p = l;
        while (l1 != null || l2 != null) {
            if (l1.val > l2.val)
                p.next = l2;
            else
                p.next = l1;
            p = p.next;
        }
        p.next = l2 == null ? l1 : l2;
        return l.next;
    }

    //54
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> nums = new ArrayList<>();
        int m = matrix.length;
        int n = matrix[0].length;
        int length = m * n;
        int i = 0, j = 0;
        int t = m;
        while (length > 0) {
            while (j < n && length > 0) {
                nums.add(matrix[i][j++]);
                length--;
            }
            i++;
            j--;
            while (i < m && length > 0) {
                nums.add(matrix[i++][j]);
                length--;
            }
            i--;
            j--;
            while (j >= 0 && length > 0) {
                nums.add(matrix[i][j--]);
                length--;
            }
            j++;
            System.out.println(i--);
            while (i > t - m && length > 0) {
                nums.add(matrix[i--][j]);
                length--;
            }
            i++;
            j++;
            m--;
            n--;
        }

        return nums;
    }

    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        String ans = "0";
        int m = num1.length(), n = num2.length();
        for (int i = n - 1; i >= 0; i--) {
            StringBuffer curr = new StringBuffer();
            int add = 0;
            for (int j = n - 1; j > i; j--) {
                curr.append(0);
            }
            int y = num2.charAt(i) - '0';
            for (int j = m - 1; j >= 0; j--) {
                int x = num1.charAt(j) - '0';
                int product = x * y + add;
                curr.append(product % 10);
                add = product / 10;
            }
            if (add != 0) {
                curr.append(add % 10);
            }
            System.out.println(curr);
            ans = addStrings(ans, curr.reverse().toString());
        }
        return ans;
    }

    public String addStrings(String num1, String num2) {
        int i = num1.length() - 1, j = num2.length() - 1, add = 0;
        StringBuffer ans = new StringBuffer();
        while (i >= 0 || j >= 0 || add != 0) {
            int x = i >= 0 ? num1.charAt(i) - '0' : 0;
            int y = j >= 0 ? num2.charAt(j) - '0' : 0;
            int result = x + y + add;
            ans.append(result % 10);
            add = result / 10;
            i--;
            j--;
        }
        ans.reverse();
        return ans.toString();
    }

    //
    public long gridGame(int[][] grid) {
        long ans = Long.MAX_VALUE;
        int n = grid[0].length;
        long[][] f = new long[2][n];
        //计算前缀和
        f[0][0] = grid[0][0];
        f[1][0] = grid[1][0];
        for (int i = 1; i < n; i++) {
            f[0][i] = f[0][i - 1] + grid[0][i];
            f[1][i] = f[1][i - 1] + grid[1][i];
        }
        //最小值
        for (int i = 0; i < n; i++) {
            if (i == 0)
                ans = f[0][n - 1];
            else
                ans = Math.min(ans, Math.max(f[0][n - 1] - f[0][i], f[1][i - 1]));
        }
        return ans;
    }

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new LinkedList<TreeNode>();
        }
        return generateTrees(1, n);
    }

    public List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> allTrees = new LinkedList<TreeNode>();
        if (start > end) {
            allTrees.add(null);
            return allTrees;
        }

        // 枚举可行根节点 1, 2, 3
        for (int i = start; i <= end; i++) {
            System.out.println("调用generateTrees : (" + start + ", " + end + ")");
            System.out.println("左子树为 : 调用generateTrees : (" + start + ", " + (i - 1) + ")");
            System.out.println("右子树为 : 调用generateTrees : (" + (i + 1) + ", " + end + ")");
            // 获得所有可行的左子树集合
            List<TreeNode> leftTrees = generateTrees(start, i - 1);
            // 获得所有可行的右子树集合
            List<TreeNode> rightTrees = generateTrees(i + 1, end);

            // 从左子树集合中选出一棵左子树，从右子树集合中选出一棵右子树，拼接到根节点上
            for (TreeNode left : leftTrees) {
                for (TreeNode right : rightTrees) {
                    TreeNode currTree = new TreeNode(i);
                    currTree.left = left;
                    currTree.right = right;
                    allTrees.add(currTree);
                }
            }
        }
        return allTrees;
    }

    //859
    public boolean buddyStrings(String s, String goal) {
        int m = s.length(), n = goal.length(), k = 0;
        if (m != n)
            return false;
        char[][] f = new char[2][2];
        for (int i = 0; i < m; i++) {
            if (s.charAt(i) != goal.charAt(i)) {
                f[k][0] = s.charAt(i);
                f[k++][1] = goal.charAt(i);
            }
        }
        if (s.equals(goal)) {
            for (int i = 0; i < m - 1; i++) {
                for (int j = i + 1; j < m; j++)
                    if (s.charAt(i) == s.charAt(j))
                        return true;
            }
        }
        if ((k == 2 && f[0][0] == f[1][1] && f[0][1] == f[1][0]))
            return true;
        return false;
    }

    //423
    public String originalDigits(String s) {
        StringBuilder sb = new StringBuilder();
        String[] nums = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};
        String[] f = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        HashMap<Character, Integer> alpha = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            if (alpha.containsKey(s.charAt(i)))
                alpha.put(s.charAt(i), alpha.get(s.charAt(i)) + 1);
            else
                alpha.put(s.charAt(i), 1);
        }
        for (int i = 0; i < nums.length; i++) {
            String str = nums[i];
            boolean flag = true;
            while (flag) {
                for (int j = 0; j < str.length(); j++) {
                    if (alpha.containsKey(str.charAt(j)) && alpha.get(str.charAt(j)) > 0)
                        flag = true;
                    else {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    for (int j = 0; j < str.length(); j++)
                        alpha.put(str.charAt(j), alpha.get(str.charAt(j)) - 1);
                    sb.append(f[i]);
                }
            }

        }
        return sb.toString();
    }

    // 完全二叉树的节点个数
    public int countNodes(TreeNode root) {
        if (root == null) return 0;
        int left = countLevel(root.left);
        int right = 0;
        int count = 0;
        while (root != null) {
            right = countLevel(root.right);

            if (left == right) {
                count += (1 << left); // 加上左边的满二叉树
                root = root.right;
            }

            if (left > right) {
                count += (1 << right);
                root = root.left;
            }
            left--;
        }
        return count;
    }

    public int countLevel(TreeNode root) {
        int level = 0;
        while (root != null) {
            root = root.left;
            level++;
        }
        return level;
    }

    List<String> res = new ArrayList<>();

    public List<String> binaryTreePaths(TreeNode root) {
        List<Integer> path = new ArrayList<>();
        traceBack(root, path);
        return res;
    }

    public void traceBack(TreeNode node, List<Integer> path) {
        path.add(node.val);

        if (node.left == null && node.right == null) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < path.size() - 1; i++) {
                sb.append(path.get(i) + "->");
            }
            sb.append(path.get(path.size() - 1));
            res.add(sb.toString());
            return;
        }

        if (node.left != null) {
            traceBack(node.left, path);
            path.remove(path.size() - 1);
        }

        if (node.right != null) {
            traceBack(node.right, path);
            path.remove(path.size() - 1);
        }
    }

    // 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        return depth(root) != 1;
    }

    // 深度
    public int depth(TreeNode node) {
        if (node == null) return 0;

        int left = depth(node.left);
        if (left == -1) return -1;

        int right = depth(node.right);
        if (right == -1) return -1;

        if (Math.abs(left - right) > 1) return -1;
        ;

        return Math.max(left, right) + 1;
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null)
            return true;
        if (p == null || q == null)
            return false;
        return p.val == q.val && isSameTree(p.left, q.right) && isSameTree(p.right, q.left);
    }


    // 左叶子之和
    int sum = 0;

    public int sumOfLeftLeaves(TreeNode root) {
        if (root != null) {
            if (isLeftLeaf(root)) {
                sum += root.left.val;
            }
            sumOfLeftLeaves(root.left);
            sumOfLeftLeaves(root.right);
        }

        return sum;
    }

    public boolean isLeftLeaf(TreeNode root) {
        if (root.left != null && root.left.left == null && root.left.right == null) {
            return true;
        }
        return false;
    }

    // 找树左下角的值
    public int findBottomLeftValue(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> val = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.poll();
                System.out.println(node.val);
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
                val.add(node.val);
            }
            res.add(new ArrayList<>(val));
        }
        return res.get(res.size() - 1).get(0);
    }

    // 路径总和2
    List<List<Integer>> myRes = new ArrayList<>();
    List<Integer> path = new ArrayList<>();

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        traceBack(root, targetSum);
        return myRes;
    }

    public void traceBack(TreeNode node, int sum) {
        path.add(node.val);
        if (node.val == sum && node.left == null && node.right == null) {
            myRes.add(new ArrayList<>(path));
            return;
        }
        if (node.left != null) {
            traceBack(node.left, sum - node.val);
            path.remove(path.size() - 1);
        }

        if (node.right != null) {
            traceBack(node.right, sum - node.val);
            path.remove(path.size() - 1);
        }

    }

    // 从前序和后序遍历构造二叉树
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder.length == 0 && inorder.length == 0)
            return null;

        TreeNode root = new TreeNode(preorder[0]);
        int index = 0;
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == root.val) {
                index = i;
                break;
            }
        }
        int[] preorder1 = Arrays.copyOfRange(preorder, 1, 1 + index);
        int[] preorder2 = Arrays.copyOfRange(preorder, 1 + index, preorder.length);

        int[] inorder1 = Arrays.copyOfRange(inorder, 0, index);
        int[] inorder2 = Arrays.copyOfRange(inorder, index, inorder.length);

        root.left = buildTree(preorder1, inorder1);
        root.right = buildTree(preorder2, inorder2);
        return root;
    }

    // 最大二叉树
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return null;
    }

    public TreeNode maxTree(int[] nums, int l, int r) {
        if (l > r) {
            return null;
        }

        int index = findMax(nums, l, r);
        TreeNode root = new TreeNode(nums[index]);
        root.left = maxTree(nums, l, index - 1);
        root.right = maxTree(nums, index + 1, r);
        return root;
    }

    //找最大值的索引
    public int findMax(int[] nums, int l, int r) {
        int max = Integer.MIN_VALUE, maxIndex = l;
        for (int i = l; i <= r; i++) {
            if (max < nums[i]) {
                max = nums[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }


    // 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null)
            return null;
        if (root1 == null || root2 == null)
            return root1 == null ? root2 : root1;
        TreeNode root = new TreeNode(root1.val + root2.val);
        root.left = mergeTrees(root1.left, root2.left);
        root.right = mergeTrees(root1.right, root2.right);
        return root;
    }

    // 不同路径2
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;

        int[][] f = new int[m][n];
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 1)
                break;
            f[i][0] = 1;
        }

        for (int i = 0; i < n; i++) {
            if (obstacleGrid[0][i] == 1)
                break;
            f[0][i] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) continue;
                f[i][j] = f[i - 1][j] + f[i][j - 1];
            }
        }
        return f[m - 1][n - 1];
    }

    public TreeNode searchBST(TreeNode root, int val) {
        if (root == null)
            return null;
        while (root != null) {
            if (root.val == val)
                return root;
            if (root.val < val)
                root = root.right;
            else
                root = root.left;
        }
        return root;
    }

    // 验证二叉搜索树
    public boolean isValidBST(TreeNode root) {
        double minValue = -Double.MIN_VALUE;

        Stack<TreeNode> stack = new Stack<>();

        while (root != null || !stack.empty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (root.val <= minValue)
                return false;

            minValue = root.val;
            root = root.right;
        }
        return true;
    }

    // 二叉搜索树的最小绝对差
    TreeNode pre;
    int result = Integer.MAX_VALUE;

    public int getMinimumDifference(TreeNode root) {

        if (root == null)
            return 0;
        traversal(root);
        return result;

    }

    private void traversal(TreeNode root) {
        if (root == null) return;

        traversal(root.left);

        if (pre != null)
            result = Math.min(result, root.val - pre.val);

        traversal(root.right);
    }

    // 串联所有单词的子串
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        // 单词长度
        int len = words.length;
        // 窗口大小
        int win = len * words[0].length();
        int start = 0, end = win;
        Map<String, Integer> map = new HashMap<>();

        for (String str : words) {
            if (map.containsKey(str))
                map.put(str, map.get(str) + 1);
            else
                map.put(str, 1);
        }

        while (start <= s.length() - win) {
            String str = s.substring(start, end);
            if (isContains(str, new HashMap<>(map), words[0].length()))
                res.add(start);
            start++;
            end++;
        }
        return res;

    }

    public boolean isContains(String str, HashMap<String, Integer> map, int len) {
        for (int i = 0; i <= str.length() - len; i = i + len) {
            String substring = str.substring(i, i + len);
            System.out.println(substring);
            if (map.containsKey(substring)) {
                map.put(substring, map.get(substring) - 1);
            } else
                return false;
        }
        for (Map.Entry entry : map.entrySet()) {
            int tag = (int) entry.getValue();
            if (tag <= 0)
                return false;
        }
        return true;
    }

    // 二叉搜索树中的众数

    List<Integer> resList = new ArrayList<>();
    int count;
    int maxCount;

    public int[] findMode(TreeNode root) {
        int[] res = new int[resList.size()];
        for (int i = 0; i < res.length; i++)
            res[i] = resList.get(i);
        return res;
    }

    public void find(TreeNode node) {

        if (node == null) return;

        find(node.left);

        if (pre == null)
            count = 1;
        else if (pre.val == node.val)
            count++;
        else
            count = 1;

        if (count == maxCount)
            resList.add(node.val);

        if (count > maxCount) {
            maxCount = count;
            resList.clear();
            resList.add(node.val);
        }

        pre = node;
        find(node.right);


    }

    // 二叉搜索树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        return null;
    }

    // 二叉搜索树中的插入操作
    public TreeNode insertIntoBST(TreeNode root, int val) {
        TreeNode node = root;
        TreeNode newNode = new TreeNode(val);
        if (root == null)
            return newNode;
        while (node != null) {
            if (node.val > val) {
                if (node.left == null) {
                    node.left = newNode;
                    return root;
                }
                node = node.left;
            }
            if (node.val < val) {
                if (node.right == null) {
                    node.right = newNode;
                    return root;
                }
                node = node.right;
            }
        }
        return root;
    }

    // 删除二叉搜索树中的节点
    public TreeNode deleteNode(TreeNode root, int key) {

        return root;
    }

    public TreeNode delete(TreeNode root, int key) {
        if (root == null)
            return null;
        if (root.val > key)
            root.left = delete(root.left, key);
        else if (root.val < key)
            root.right = delete(root.right, key);
        else {
            if (root.left == null || root.right == null)
                return root.left == null ? root.right : root.left;
            else {
                // 将左子树放在右子树的最坐下结点
                TreeNode right = root.right;
                while (right.left != null) {
                    right = right.left;
                }
                right.left = root.left;
                root = root.right;
                return root;
            }
        }
        return root;
    }


    // 修剪二叉搜索树
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null)
            return null;
        if (root.val < low || root.val > high) {
            if (root.left == null || root.right == null)
                return root.left == null ? root.right : root.left;
            else {
                // 将左子树放在右子树的最坐下结点
                TreeNode right = root.right;
                while (right.left != null) {
                    right = right.left;
                }
                right.left = root.left;
                root = root.right;
                return root;
            }
        }
        root.left = trimBST(root.left, low, high);
        root.left = trimBST(root.right, low, high);
        return root;
    }

    // 斐波那契数
    public int fib(int n) {
        int f[] = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            if (i == 0) f[i] = 0;
            else if (i == 1) f[i] = 1;
            else f[i] = f[i - 1] + f[i - 2];
        }
        return f[n];
    }

    // 爬楼梯
    public int climbStairs(int n) {
        int[] f = new int[n];
        for (int i = 0; i < n; i++) {
            if (i == 0) f[i] = 1;
            else if (i == 1) f[i] = 2;
            else f[i] = f[i - 1] + f[i - 2];
        }
        return f[n - 1];
    }

    // 使用最小花费爬楼梯
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        int[] f = new int[n + 1];
        f[0] = 0;
        f[1] = 0;
        for (int i = 2; i <= n; i++) {
            f[i] = Math.min(f[i - 1] + cost[i - 1], f[i - 2] + cost[i - 2]);
        }
        return f[n];
    }


    // 整数拆分
    public int integerBreak(int n) {
        int[] f = new int[n + 1];
        f[2] = 1;
        for (int i = 3; i <= n; i++) {
            int max = Integer.MIN_VALUE;
            for (int j = i - 1; j >= 2; j--) {
                max = Math.max(Math.max(f[j] * (i - j), max), j * (i - j));
            }
            f[i] = max;
        }
        return f[n];
    }


    // 不同的二叉搜索树
    public int numTrees(int n) {
        int[] f = new int[n + 1];
        f[0] = 1;
        f[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++)
                f[i] += f[j] * f[i - j - 1];
        }
        return f[n];
    }


    // 二分查找
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] > target) right = mid;
            else left = mid + 1;
        }
        return -1;
    }


    // 反转链表
    public ListNode reverseList(ListNode head) {
        if (head != null) {
            ListNode next = head.next;
            ListNode tmp = null;
            head.next = tmp;
            while (next != null) {
                ListNode cur = next;
                cur.next = head;
                head = cur;
                next = next.next;
            }
        }
        return head;
    }

    // 回文子串
    public int countSubstrings(String s) {
        int n = s.length();
        int res = 0;

        int[][] f = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) f[i][j] = 1;
                else {
                    if (s.charAt(i) == s.charAt(j)) {
                        if (j - i == 1) f[i][j] = 1;
                        else if (f[i + 1][j - 1] == 1) f[i][j] = 1;
                    }
                }
                res += f[i][j];
            }
        }
        return res;
    }


    // 最长回文子序列
    public int longestPalindromeSubseq(String s) {
        int n = s.length();

        int[][] f = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) {
                    f[i][j] = 1;
                    continue;
                }
                if (s.charAt(i) == s.charAt(j))
                    f[i][j] = 2 + f[i + 1][j - 1];
                else
                    f[i][j] = Math.max(f[i][j - 1], f[i + 1][j]);
            }

        }
        return f[0][n - 1];
    }

    public String longestPalindrome(String s) {
        int n = s.length(), max = Integer.MIN_VALUE;
        String res = null;
        int[][] f = new int[n][n];
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i; j < n; j++) {
                if (i == j) f[i][j] = 1;
                else {
                    if (s.charAt(i) == s.charAt(j)) {
                        if (f[i + 1][j - 1] == 1 || j - i == 1) f[i][j] = 1;
                    }
                }
                if (j - i > max && f[i][j] == 1) {
                    max = i - j;
                    res = s.substring(i, j + 1);
                }
            }
        }
        return res;
    }

    // 两数之和
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++)
            map.put(nums[i], i);
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i]) && map.get(target - nums[i]) != i)
                return new int[]{i, map.get(target - nums[i])};
        }
        return res;
    }


    // 三数之和
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        Arrays.sort(nums);
        for (int i = 0; i < n; i++) {
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            int left = i + 1, right = n - 1;
            while (left < right) {
                if (nums[i] + nums[left] + nums[right] == 0) {
                    res.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (right > left && nums[right] == nums[right - 1]) right--;
                    while (right > left && nums[left] == nums[left + 1]) left++;
                    left++;
                    right--;
                } else if (nums[i] + nums[left] + nums[right] < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return res;
    }

    // 四数之和
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n - 1; i++) {
            if (i > 0 && nums[i] == nums[i - 1])
                continue;

            for (int j = i + 1; j < n; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1])
                    continue;

                int left = j + 1, right = n - 1;
                while (left < right) {
                    int sum = nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) left++;
                        while (left < right && nums[right] == nums[right - 1]) right--;
                        left++;
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else
                        right--;
                }
            }
        }
        return res;
    }

    // 四数相加二
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        int res = 0;
        HashMap<Integer, Integer> mp = new HashMap<>();
        for (int u : nums1) {
            for (int v : nums2) {
                mp.put(u + v, mp.getOrDefault(u + v, 0) + 1);
            }
        }

        for (int u : nums3) {
            for (int v : nums4) {
                if (mp.containsKey(-u - v))
                    res += mp.get(-u - v);
            }
        }
        return res;
    }

    // 分发饼干
    public int findContentChildren(int[] g, int[] s) {
        int res = 0;
        Arrays.sort(g);
        Arrays.sort(s);
        int start = s.length - 1;
        for (int j = g.length - 1; start >= 0 && j >= 0; j--) {
            if (s[start] >= g[j]) {
                res ++;
                start--;
            }
        }
        return res;
    }

    // 组合

    //List<List<Integer>> nums = new ArrayList<>();
    public List<List<Integer>> combine(int n, int k) {
        List<Integer> path = new ArrayList<>();
        backTrace(1, k, n, path);
        return nums;
    }
    void backTrace(int start, int len, int n, List<Integer> path) {
        if (path.size() == len) {
            nums.add(new ArrayList(path));
            return;
        }
        for (int i = start; i <= n - (len - path.size()) + 1; i++) {
            path.add(i);
            backTrace(i + 1, len, n, path);
            path.remove(path.size() - 1);
        }
    }
    static void backTrace1(List<Integer> path, int n, int k, int start, int sum) {
        if (path.size() == k) {
            if (sum == n)
                nums.add(new ArrayList(path));
            return;
        }
        for (int i = start; i <= n - (k - path.size()) + 1; i++) {
            path.add(i);
            sum += i;
            backTrace1(path, n, k, i + 1, sum);
            sum -= path.get(path.size() - 1);
            path.remove(path.size() - 1);
        }
    }

    // 组合总和3
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<Integer> path = new ArrayList<>();
        backTrace1(path, n, k, 1, 0);
        return nums;
    }
    // 摆动序列
    public int wiggleMaxLength(int[] nums) {
        int n =  nums.length;
        int res = 1;
        if (n <= 1) return 1;
        int cur = 0;
        int pre = 0;
        for (int i = 0; i < n - 1; i++) {
            cur = nums[i + 1] - nums[i];
            if ((cur > 0 && pre < 0) || (cur < 0 && pre > 0)) {
                res++;
                pre = cur;
            }
        }

        return res;
    }

    // 最大子数组和
    public int maxSubArray(int[] nums) {
        int res = Integer.MIN_VALUE;
        int count = 0;

        for (int i = 0; i < nums.length; i++) {
            count += nums[i];
            if (count > res) {
                res = count;
            }
            if (count < 0) {
                count = 0;
            }
        }
        return res;
    }


ReentrantReadWriteLock

}
