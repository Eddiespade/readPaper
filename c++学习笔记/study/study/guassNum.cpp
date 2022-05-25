#include <iostream>
using namespace std;
// time 系统时间头文件包含
#include <ctime>

int GuassNum()
{	
	// 添加随机数种子
	srand((unsigned int)time(NULL));
	int num = rand() % 100 + 1;
	while (true)
	{
		int a = 0;
		cout << "请猜数字：";
		cin >> a;
		if (a < num)
		{
			cout << "你猜的数字太小了，请重猜；" << endl;
		}
		else if (a > num) 
		{
			cout << "你猜的数字太大了，请重猜；" << endl;
		}
		else
		{
			cout << "恭喜你猜对了；正确的数字为：" << num << endl;
			break;
		}
	}

	system("pause");
	return 0;
}