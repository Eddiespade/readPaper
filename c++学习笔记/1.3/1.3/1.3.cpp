#include <iostream>

#define g 3

int main()
{
    /* 常量与变量的起名：
        1. 不能重名 
        2. 不能和关键字重 
        3. 必须是字母或者字母+数字组合，符号仅 _ 可用
        4. 名字不能以数字开头
    */
    int a = 1;
    std::cout << a << std::endl;
    a = 2;
    std::cout << a << std::endl;
    /* 变量的定义方式：
       1. 类型 名字{初始值}；  如 int a{2}；
       2. 类型 名字 = 初始值；  如 int a = 2；
    */
    int b{ 2 };
    std::cout << b << std::endl;
    int c  = 2;
    std::cout << c << std::endl;
    /* 常量的定义方式：
       1. 直接用值 如 5；
       2. const 类型 名字{初始值}；  如 const int a{2}；
       3. 类型 const 名字{初始值}；  如 int const a{2}；
       4. # define 名字 值；         如 #define a 2；
       注：常量不能赋值！
    */
    const int d{ 3 };
    std::cout << d << std::endl;
    int const e = 3;
    std::cout << e << std::endl;
    std::cout << g << std::endl;
    
}
