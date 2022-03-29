package com.yang;

import java.util.Date;
import java.util.HashMap;
import java.util.Map;

class LRUCache {
    class DLinkedNode{
        int key;
        int value;
        DLinkedNode pre;
        DLinkedNode next;

        public DLinkedNode() {
        }

        public DLinkedNode(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }
    private Map<Integer, DLinkedNode> cache = new HashMap<>();
    private int size;
    private int capacity;
    private DLinkedNode head, tail;

    public LRUCache(int capacity) {
        this.size = 0;
        this.capacity = capacity;
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head.next = tail;
        tail.pre = head;
    }
    public int get(int key) {
        DLinkedNode node = cache.get(key);
        if(node != null){
            moveToHead(node);
            return node.value;
        }
        return -1;
    }

    private void moveToHead(DLinkedNode node) {
        node.next.pre = node.pre;
        node.pre.next = node.next;
        node.next = head.next;
        node.next.pre = node;
        head.next = node;
        node.pre = head;
    }

    public void put(int key, int value) {
        DLinkedNode node = cache.get(key);
        if(node == null){
            DLinkedNode newNode = new DLinkedNode(key, value);
            cache.put(key, newNode);
            addToHead(newNode);
            ++size;
            if(size > capacity){
                DLinkedNode tail = removeTail();
                cache.remove(tail.key);
                --size;
            }
        }else {
            node.value = value;
            moveToHead(node);
        }
    }

    private DLinkedNode removeTail() {
        DLinkedNode res = tail.pre;
        tail.pre = tail.pre.pre;
        tail.pre.next = tail;
        return res;
    }

    private void addToHead(DLinkedNode node) {
        node.next = head.next;
        node.next.pre = node;
        head.next = node;
        node.pre = head;
    }

}