#include <iostream>
using namespace std;

int ThirdPig()
{
	// ��ֻС������أ��ж���֪����
	int a, b, c;
	cout << "��������ֻ�������" << endl;
	cin >> a >> b >> c;
	int max_ab = 0;
	if (a > b) max_ab = a;
	else max_ab = b;
	int max_c = 0;
	if (c > max_ab) max_c = c;
	else max_c = max_ab;
	cout << "���ص�����Ϊ��" << max_c << endl;

	system("pause");
	return 0;

}