1. RTX 3800 L2 cache size : 5MB -> allocate 10MB buffer
2. L2 cache line 128B

# steps
1. allocate 10MB buffer
2. 每隔128B取一个地址，首先使用前16个地址填充一个Eviction_set
3. 利用算法1，遍历buffer中所有地址找到一组set
4. 换地址找到所有sets

# general data structure
双层链表，每个eviction set用链表表示，最后将链表转化成数组
```
____________     ____________ 
|evcit  set1|    |           |
| addr->addr|    |           |
|           | -> |           |
|           |    |           |
|___________|    |___________| 
```

# Specific
已知3080可以找到2560个eviction sets，直接用2560 * 16的数组保存地址即可，多余的地址从地址表中删除即可。