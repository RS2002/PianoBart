旋律提取（melody extract）任务有两个子任务：

1. simplified：分辨melody/non-melody，数据存放在`POP909/simplified`中

   ```json
   {
       "MELODY": 1,
       "BRIDGE": 0,
       "PIANO": 0,
       "OTHER": 2 // OTHER中包括PAD/EOS/CAU
   }
   ```

   

2. default：分辨melody/bridge/accompaniment，数据存放在`POP909/default`中

   ```json
   {
       "MELODY": 0,
       "BRIDGE": 1,
       "PIANO": 2,
       "OTHER": 3 // OTHER中包括PAD/EOS/CAU
   }
   ```

   