# 2016MachineLearningFinalProject

##Data Set Description
### test_k.cv , train_k.cv
增加三種新 features，分別為第 19, 20, 21 維的資料

1. 此 `enrollment id` 登入 `problem` 這種功能的次數：

    原因：猜測登入`problem`功能越多次的學生越認真，越不會 drop

2. 此 `enrollment id` 登入 `discussion` 這種功能的次數。

    原因：猜測登入`discussion`功能越多次的學生也越認真，越不會 drop
3. 此 `enrollment id` 修的 `course` 裡面所含 `chapter` 的數量，即章節數。

    原因：猜測修的課章節越多，學生會覺得越不耐煩，導致不容易修完而 drop