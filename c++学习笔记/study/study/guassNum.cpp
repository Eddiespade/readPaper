#include <iostream>
using namespace std;
// time ϵͳʱ��ͷ�ļ�����
#include <ctime>

int GuassNum()
{	
	// ������������
	srand((unsigned int)time(NULL));
	int num = rand() % 100 + 1;
	while (true)
	{
		int a = 0;
		cout << "������֣�";
		cin >> a;
		if (a < num)
		{
			cout << "��µ�����̫С�ˣ����ز£�" << endl;
		}
		else if (a > num) 
		{
			cout << "��µ�����̫���ˣ����ز£�" << endl;
		}
		else
		{
			cout << "��ϲ��¶��ˣ���ȷ������Ϊ��" << num << endl;
			break;
		}
	}

	system("pause");
	return 0;
}