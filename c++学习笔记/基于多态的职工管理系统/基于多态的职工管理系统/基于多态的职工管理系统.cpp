#include <iostream>
using namespace std;
#include <string>
#include "workerManager.h"


int main()
{
	WorkerManager wm;
	int choice = 0;
	while (true)
	{
		//չʾ�˵�
		wm.Show_Menu();
		cout << "����������ѡ��:" << endl;
		cin >> choice;

		switch (choice)
		{
		case 0: //�˳�ϵͳ
			wm.ExitSystem();
			break;
		case 1: //���ְ��
			break;
		case 2: //��ʾְ��
			break;
		case 3: //ɾ��ְ��
			break;
		case 4: //�޸�ְ��
			break;
		case 5: //����ְ��
			break;
		case 6: //����ְ��
			break;
		case 7: //����ļ�
			break;
		default:
			system("cls");
			break;
		}
	}

	system("pause");
	return 0;
}