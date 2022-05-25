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
	// Ã°ÅİÅÅĞò
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
		{ "Áõ±¸",23,"ÄĞ"},
		{ "¹ØÓğ",22,"ÄĞ"},
		{ "ÕÅ·É",20,"ÄĞ"},
		{ "ÕÔÔÆ",21,"ÄĞ"},
		{ "õõ²õ",19,"Å®"},
	};
	
	// ¶ÔÓ¢ĞÛÅÅĞò
	popSorted(heros);

	// ´òÓ¡Ã¿¸öÓ¢ĞÛ
	for (int i = 0; i < 5; i++)
	{
		cout << heros[i].name << heros[i].age << heros[i].sex << endl;
	}

	system("pause");
	return 0;
}