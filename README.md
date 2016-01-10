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

##Feature Importance
Importance: "How many times do DTrees use the feature to seperate data"

0 ID 65113
1 user_log_num 40401
2 course_log_num 18083
3 take_course_num 12446
4 take_user_num 13542
5 log_num 78409
6 server_nagivate 21700
7 server_access 19527
8 server_problem 10630
9 browser_access 19833
10 browser_problem 14883
11 browser_page_close 20269
12 browser_video 18421
13 server_discussion 20673
14 server_wiki 10529
15 chapter_count 14931
16 sequential_count 20124
17 video_count 13395
18 problem_count 9120
19 discussion_count 0
20 video_count 13395
21 drop_rate 0
