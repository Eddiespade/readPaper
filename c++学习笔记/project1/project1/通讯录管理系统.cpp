#include <iostream>
#include <string>
using namespace std;
#define MAX 1000


//��ϵ�˽ṹ��
struct Person
{
	string name;				//��ϵ������
	int sex;					//��ϵ���Ա�1-�С�2-Ů
	int age;					//��ϵ������
	string phone;				//��ϵ�˵绰
	string addr;				//��ϵ��סַ
};

//ͨѶ¼�ṹ��
struct AddressBook
{
	Person personArray[MAX];	//ͨѶ¼�б������ϵ�����飬���1000��
	int len;					//ͨѶ¼����Ա����
};

//��ʾ�˵�����
void showMenu()
{
	cout << "*************************" << endl;
	cout << "***** 1�������ϵ�� *****" << endl;
	cout << "***** 2����ʾ��ϵ�� *****" << endl;
	cout << "***** 3��ɾ����ϵ�� *****" << endl;
	cout << "***** 4��������ϵ�� *****" << endl;
	cout << "***** 5���޸���ϵ�� *****" << endl;
	cout << "***** 6�������ϵ�� *****" << endl;
	cout << "***** 0���˳�ͨѶ¼ *****" << endl;
	cout << "*************************" << endl;
}

//�����ϵ�˺���
void addPerson(AddressBook * addressBook)
{
	//�ж�ͨѶ¼�����Ƿ�Ϊ��
	if (addressBook->len == MAX)
	{
		cout << "ͨѶ¼�������޷�������ӣ�" << endl;
		return;
	}
	else
	{
		//��Ӿ�����ϵ��
		//�������Ա����䣬�绰��סַ
		string name;
		cout << "������������" << endl;
		cin >> name;
		addressBook->personArray[addressBook->len].name = name;

		cout << "�������Ա�" << endl;
		cout << "1--�� 2--Ů" << endl;
		int sex = 0;
		while (true)
		{
			cin >> sex;
			if (sex == 1 || sex == 2)
			{
				addressBook->personArray[addressBook->len].sex = sex;
				break;
			}
			cout << "�����������������룡" << endl;
		}

		cout << "���������䣺" << endl;
		int age = 0;
		while (true)
		{
			cin >> age;
			if (age >= 0 && age <= 100)
			{
				addressBook->personArray[addressBook->len].age = age;
				break;
			}
			cout << "�����������������룡" << endl;
		}

		cout << "��������ϵ�绰��" << endl;
		string phone;
		cin >> phone;
		addressBook->personArray[addressBook->len].phone = phone;

		cout << "�������ͥ��ַ��" << endl;
		string address;
		cin >> address;
		addressBook->personArray[addressBook->len].addr = address;

		//����ͨѶ¼����
		addressBook->len++;

		cout << "��ӳɹ���" << endl;
		system("pause");
		system("cls"); // ��������
	}
}

//��ʾ��ϵ�˺���
void showPerson(AddressBook* addressBook)
{
	if (addressBook->len == 0)
	{
		cout << "ͨѶ¼ĿǰΪ��" << endl;
	}
	else
	{
		for (int i = 0; i < addressBook->len; i++)
		{
			cout << "������ " << addressBook->personArray[i].name;
			//�Ľ��� cout << "\t�Ա�" << (addressBook->personArray[i].sex == 1 ? "��" : "Ů");
			if (addressBook->personArray[i].sex == 1)
				cout << "\t�Ա� ��";
			else
				cout << "\t�Ա� Ů";
			cout << "\t���䣺 " << addressBook->personArray[i].age <<
				"\t��ϵ�绰�� " << addressBook->personArray[i].phone <<
				"\t��ͥסַ�� " << addressBook->personArray[i].addr << endl;
		}
	}

	system("pause");
	system("cls"); // ��������
}

// �ж���ϵ���Ƿ���ں���
int isExist(AddressBook* addressBook, string name)
{
	for (int i = 0; i < addressBook->len; i++)
	{
		if (addressBook->personArray[i].name == name)
			return i; // ����ϵ�˴��ڣ����ظ���ϵ���������е��±�
	}

	return -1; // �������������飬��ϵ�˲����ڣ�����-1
}

// ��ʾһ����ϵ�˵���Ϣ����
void showOnePerson(AddressBook* addressBook, int i)
{
	cout << "������ " << addressBook->personArray[i].name;
	cout << "\t�Ա�" << (addressBook->personArray[i].sex == 1 ? "��" : "Ů");
	cout << "\t���䣺 " << addressBook->personArray[i].age <<
		"\t��ϵ�绰�� " << addressBook->personArray[i].phone <<
		"\t��ͥסַ�� " << addressBook->personArray[i].addr << endl;
}

// ɾ����ϵ�˺���
void deletePerson(AddressBook* addressBook)
{
	cout << "��������Ҫɾ������ϵ�ˣ�" << endl;
	string name;
	cin >> name;
	int result = isExist(addressBook, name);
	if (result == -1)
	{
		cout << "���޴��ˣ�" << endl;
	}
	else
	{
		for (int i = result; i < addressBook->len; i++)
		{
			addressBook->personArray[i] = addressBook->personArray[i + 1];
		}
		addressBook->len--;
		cout << "ɾ���ɹ���" << endl;
	}

	system("pause");
	system("cls");
}

//������ϵ�˺���
void findPerson(AddressBook* addressBook)
{
	cout << "��������Ҫ���ҵ���ϵ�ˣ�" << endl;
	string name;
	cin >> name;
	int result = isExist(addressBook, name);
	if (result == -1)
	{
		cout << "���޴��ˣ�" << endl;
	}
	else
	{
		cout << "���ҳɹ���" << endl;
		showOnePerson(addressBook, result);
	}

	system("pause");
	system("cls");
}

//�޸���ϵ�˺���
void modifyPerson(AddressBook* addressBook)
{
	cout << "��������Ҫ�޸ĵ���ϵ�ˣ�" << endl;
	string name;
	cin >> name;
	int result = isExist(addressBook, name);
	if (result == -1)
	{
		cout << "���޴��ˣ�" << endl;
	}
	else
	{
		string name;
		cout << "������������" << endl;
		cin >> name;
		addressBook->personArray[result].name = name;

		cout << "�������Ա�" << endl;
		cout << "1--�� 2--Ů" << endl;
		int sex = 0;
		while (true)
		{
			cin >> sex;
			if (sex == 1 || sex == 2)
			{
				addressBook->personArray[result].sex = sex;
				break;
			}
			cout << "�����������������룡" << endl;
		}

		cout << "���������䣺" << endl;
		int age = 0;
		while (true)
		{
			cin >> age;
			if (age >= 0 && age <= 100)
			{
				addressBook->personArray[result].age = age;
				break;
			}
			cout << "�����������������룡" << endl;
		}

		cout << "��������ϵ�绰��" << endl;
		string phone;
		cin >> phone;
		addressBook->personArray[result].phone = phone;

		cout << "�������ͥ��ַ��" << endl;
		string address;
		cin >> address;
		addressBook->personArray[result].addr = address;
	}

	system("pause");
	system("cls");
}

//���������ϵ�˺���
void cleanPerson(AddressBook* addressBook)
{
	//ֻ�轫ͨѶ¼����ϵ��������Ϊ0���߼����
	addressBook->len = 0;
	cout << "ͨѶ¼����գ�" << endl;
	system("pause");
	system("cls");
}

int main()
{
	//����ͨѶ¼�ṹ�����
	AddressBook addressBook;
	//��ʼ��ͨѶ¼��Ա����
	addressBook.len = 0;

	int select = 0; // �����û�ѡ������ı���

	while (true)
	{
		// �˵�����
		showMenu();
		cin >> select;
		switch (select)
		{
		case 1: // �����ϵ��
			addPerson(&addressBook);
			break;
		case 2: //��ʾ��ϵ��
			showPerson(&addressBook);
			break;
		case 3: //ɾ����ϵ��
			deletePerson(&addressBook);
			break;
		case 4: //������ϵ��
			findPerson(&addressBook);
			break;
		case 5: //�޸���ϵ��
			modifyPerson(&addressBook);
			break;
		case 6: //�����ϵ��
			cleanPerson(&addressBook);
			break;
		case 0: // �˳�ͨѶ¼
			cout << "��ӭ�´�ʹ��" << endl;
			system("pause");
			return 0;
			break;
		default:
			break;
		}
	}

	system("pause");
	return 0;
}