# 1.1 基本知识

1.  初始化解析

   ```c++
   #include <iostream>
   
   
   // 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
   // 调试程序: F5 或调试 >“开始调试”菜单
   int main()
   {
       
       // std::cout 输出 “ ” 中的内容
       std::cout << "Hello World!\n";
       // 与下面的输出一致，注意  << 方向不能反向
       std::cout << "Hello World!" << std::endl;
       // 反斜杠\： \\ 等价于输出 \
       // 正斜杠/
   }
   ```

# 1.2 基本语法

```c++
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
```

# 1.3 常量和变量

```c++
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
```

