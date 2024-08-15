/discard/discard.cu 利用算法1来build conflictset candidate evictionset，问题在于找不到eviction set 
/discard/eviction.cu 先构建conflictset, 选取target，替换掉conflictset最后一个来判断eviciton。目前问题在于地址连续，一个考虑方向是make less noise


/timing/main.cu 利用时间差异构建conflict set和eviction set，没办法remove掉冲突的地址，而且时间也不合理。