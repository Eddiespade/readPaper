#include <iostream>
using namespace std;
#include <string>


struct Hero
{
	string name;
	int age;
	string sex;
};

void popSorted(Hero arr[]) 
{
	// ð������
	for (int i = 4; i > 0; i--) {
		for (int j = 0; j < i; j++)
		{
			if (arr[j].age > arr[j + 1].age) {
				Hero temp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = temp;
			}
		}
	}
}

int main()
{
	Hero heros[5] = {
		{ "����",23,"��"},
		{ "����",22,"��"},
		{ "�ŷ�",20,"��"},
		{ "����",21,"��"},
		{ "����",19,"Ů"},
	};
	
	// ��Ӣ������
	popSorted(heros);

	// ��ӡÿ��Ӣ��
	for (int i = 0; i < 5; i++)
	{
		cout << heros[i].name << heros[i].age << heros[i].sex << endl;
	}

	system("pause");
	return 0;
}