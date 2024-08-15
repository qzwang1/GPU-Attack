/test/test_discard.cu 是郭老师提供的测试discard的示例

/access_time 是测试memory hit， L2 hit时间的示例

/test_discard/simple.cu 利用郭老师提供的discard，结果还是不对 
/test_discard/test.cu    将mem调整为128B，discard有效，输出b[2]有效和输出时间不一致，但是print a时得到0