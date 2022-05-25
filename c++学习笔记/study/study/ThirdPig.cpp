#include <iostream>
using namespace std;

int ThirdPig()
{
	// 三只小猪称体重，判断哪知最重
	int a, b, c;
	cout << "请输入三只猪的体重" << endl;
	cin >> a >> b >> c;
	int max_ab = 0;
	if (a > b) max_ab = a;
	else max_ab = b;
	int max_c = 0;
	if (c > max_ab) max_c = c;
	else max_c = max_ab;
	cout << "最重的体重为：" << max_c << endl;

	system("pause");
	return 0;

}