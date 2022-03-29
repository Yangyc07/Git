package com.yang;

import java.util.Random;

public class Main {

    public static void main(String[] args) {
	// write your code here
        int nums [] = {5, 2, 3, 1};
        new Main().quickSort(nums, 0, nums.length - 1);
        for(int i = 0; i < nums.length; i++)
            System.out.println(nums[i]);
    }

    public int []sort(int []nums, int low, int high){
        if(low < high){
            int mid = low + (high - low) / 2;
            sort(nums, low, mid);
            sort(nums, mid + 1, high);
            merge(nums, low, mid, high);
        }
        return nums;
    }
    public void merge(int [] nums, int low, int mid, int high){
        int [] temp = new int [high - low + 1];
        int i = low;
        int j = mid + 1;
        int k = 0;
        while(i <= mid && j <= high){
            if(nums[i] < nums[j])
                temp[k++] = nums[i++];
            else
                temp[k++] = nums[j++];
        }
        while(i <= mid)
            temp[k++] = nums[i++];
        while(j <= high)
            temp[k++] = nums[j++];
        for(int x = 0; x < temp.length; x++)
            nums[x + low] = temp[x];
    }

    //最大堆排序
    public void sort(int[] nums){
        //建堆 [1, n]
        int n = nums.length;
        for(int i = n / 2 - 1; i >= 0; i--)
            sink(nums, i, n - 1);
        for(int i = n - 1; i >= 0; i--){
            //交换
            int temp = nums[i];
            nums[i] = nums[0];
            nums[0] = temp;
            sink(nums, 0, i - 1);//前i-1个数据
        }
    }
    public void sink(int []nums, int i, int n){//下沉，保持堆的性质 [1 - n];
        while(2 * i + 1 <= n){
            int largest =  i;
            int left = 2 * i + 1;
            int right = left + 1;
            if(left <= n && nums[left] > nums[largest])
                largest = left;
            if(right <= n && nums[right] > nums[largest])
                largest = right;
            if(largest != i) {
                int temp = nums[i];
                nums[i] = nums[largest];
                nums[largest] = temp;
                i = largest;
            }else
                break;
        }
    }
    //1550
    public boolean threeConsecutiveOdds(int[] arr) {
        int n = arr.length;
        if(n < 3)
            return false;
        int i = 0;
        while(i < n - 3){
            if(arr[i] % 2 == 1 && arr[i + 1] % 2 == 1 && arr[i + 2] % 2 == 1)
                return true;
            i++;
        }
        return false;
    }
    //quickSort
    public void quickSort(int nums[], int p, int r){
        if(p < r){
            int q = partion(nums, p, r);
            quickSort(nums,p, q - 1);
            quickSort(nums,q + 1, r);
        }
    }
    public int partion(int nums[], int p, int r){ //[p, r]
        int ran = randomNumber(p, r);
        int temp = nums[ran];
        nums[ran] = nums[r];
        nums[r] = temp;

        int i = p - 1;
        int x = nums[r];
        for(int j = p; j <= r - 1; j++){
            if(nums[j] < x){
                i++;
                temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
            }
        }

        i++;
        temp = nums[i];
        nums[i] = nums[r];
        nums[r] = temp;
        return i;
    }
    public static int randomNumber(int minNum,int maxNum){
        Random rand = new Random();
        int randomNum = rand.nextInt(maxNum);
        randomNum = randomNum%(maxNum-minNum+1)+minNum;
        return randomNum;
    }
}
