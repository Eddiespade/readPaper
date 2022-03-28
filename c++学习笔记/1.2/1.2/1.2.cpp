/* # 为预处理指令 <iostream> 库函数*/

#include <iostream>

/* 以main开始，以main函数结束*/
int main()
{
    // 大小写严格，不可以乱用
    /* printf 和 std::cout 的区别：
        printf      函数
        std::cout 	类
    */
    std::cout << "Hello World!\n";
    printf("你好， \tc++");
    // system 系统关机命令
    system("shutdown /s");
    // system 取消系统关机命令
    system("shutdown /a");
    // 暂停
    system("pause");
    system("pause已被成功完成！");
    // 清屏
    system("cls");
    // 更改颜色背景
    system("color fc");
}