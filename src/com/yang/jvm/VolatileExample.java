package com.yang.jvm;

public class VolatileExample extends Thread{
    public int a = 0;
    volatile boolean flag = false;

    @Override
    public void run() {
        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        a = 1;
        flag = true;
    }

    public int getFlag() {
        return a;
    }
}

class testVolatile {

    public static void main(String[] args) {
        VolatileExample thread = new VolatileExample();
        thread.start(); // 启动

        while (true) {
            if (thread.flag) {
                System.out.println("flag is : " + thread.flag);
                System.out.println("a is : " + thread.a);
            }
        }
    }
}