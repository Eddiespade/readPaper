#include <iostream>
#include <string>
using namespace std;
#define MAX 1000


//联系人结构体
struct Person
{
	string name;				//联系人姓名
	int sex;					//联系人性别，1-男、2-女
	int age;					//联系人年龄
	string phone;				//联系人电话
	string addr;				//联系人住址
};

//通讯录结构体
struct AddressBook
{
	Person personArray[MAX];	//通讯录中保存的联系人数组，最多1000人
	int len;					//通讯录中人员个数
};

//显示菜单函数
void showMenu()
{
	cout << "*************************" << endl;
	cout << "***** 1、添加联系人 *****" << endl;
	cout << "***** 2、显示联系人 *****" << endl;
	cout << "***** 3、删除联系人 *****" << endl;
	cout << "***** 4、查找联系人 *****" << endl;
	cout << "***** 5、修改联系人 *****" << endl;
	cout << "***** 6、清空联系人 *****" << endl;
	cout << "***** 0、退出通讯录 *****" << endl;
	cout << "*************************" << endl;
}

//添加联系人函数
void addPerson(AddressBook * addressBook)
{
	//判断通讯录人数是否为满
	if (addressBook->len == MAX)
	{
		cout << "通讯录已满，无法继续添加！" << endl;
		return;
	}
	else
	{
		//添加具体联系人
		//姓名，性别，年龄，电话，住址
		string name;
		cout << "请输入姓名：" << endl;
		cin >> name;
		addressBook->personArray[addressBook->len].name = name;

		cout << "请输入性别：" << endl;
		cout << "1--男 2--女" << endl;
		int sex = 0;
		while (true)
		{
			cin >> sex;
			if (sex == 1 || sex == 2)
			{
				addressBook->personArray[addressBook->len].sex = sex;
				break;
			}
			cout << "输入有误！请重新输入！" << endl;
		}

		cout << "请输入年龄：" << endl;
		int age = 0;
		while (true)
		{
			cin >> age;
			if (age >= 0 && age <= 100)
			{
				addressBook->personArray[addressBook->len].age = age;
				break;
			}
			cout << "输入有误！请重新输入！" << endl;
		}

		cout << "请输入联系电话：" << endl;
		string phone;
		cin >> phone;
		addressBook->personArray[addressBook->len].phone = phone;

		cout << "请输入家庭地址：" << endl;
		string address;
		cin >> address;
		addressBook->personArray[addressBook->len].addr = address;

		//更新通讯录人数
		addressBook->len++;

		cout << "添加成功！" << endl;
		system("pause");
		system("cls"); // 清屏操作
	}
}

//显示联系人函数
void showPerson(AddressBook* addressBook)
{
	if (addressBook->len == 0)
	{
		cout << "通讯录目前为空" << endl;
	}
	else
	{
		for (int i = 0; i < addressBook->len; i++)
		{
			cout << "姓名： " << addressBook->personArray[i].name;
			//改进： cout << "\t性别：" << (addressBook->personArray[i].sex == 1 ? "男" : "女");
			if (addressBook->personArray[i].sex == 1)
				cout << "\t性别： 男";
			else
				cout << "\t性别： 女";
			cout << "\t年龄： " << addressBook->personArray[i].age <<
				"\t联系电话： " << addressBook->personArray[i].phone <<
				"\t家庭住址： " << addressBook->personArray[i].addr << endl;
		}
	}

	system("pause");
	system("cls"); // 清屏操作
}

// 判断联系人是否存在函数
int isExist(AddressBook* addressBook, string name)
{
	for (int i = 0; i < addressBook->len; i++)
	{
		if (addressBook->personArray[i].name == name)
			return i; // 该联系人存在，返回该联系人在数组中的下标
	}

	return -1; // 遍历完整个数组，联系人不存在，返回-1
}

// 显示一个联系人的信息函数
void showOnePerson(AddressBook* addressBook, int i)
{
	cout << "姓名： " << addressBook->personArray[i].name;
	cout << "\t性别：" << (addressBook->personArray[i].sex == 1 ? "男" : "女");
	cout << "\t年龄： " << addressBook->personArray[i].age <<
		"\t联系电话： " << addressBook->personArray[i].phone <<
		"\t家庭住址： " << addressBook->personArray[i].addr << endl;
}

// 删除联系人函数
void deletePerson(AddressBook* addressBook)
{
	cout << "请输入您要删除的联系人：" << endl;
	string name;
	cin >> name;
	int result = isExist(addressBook, name);
	if (result == -1)
	{
		cout << "查无此人！" << endl;
	}
	else
	{
		for (int i = result; i < addressBook->len; i++)
		{
			addressBook->personArray[i] = addressBook->personArray[i + 1];
		}
		addressBook->len--;
		cout << "删除成功！" << endl;
	}

	system("pause");
	system("cls");
}

//查找联系人函数
void findPerson(AddressBook* addressBook)
{
	cout << "请输入您要查找的联系人：" << endl;
	string name;
	cin >> name;
	int result = isExist(addressBook, name);
	if (result == -1)
	{
		cout << "查无此人！" << endl;
	}
	else
	{
		cout << "查找成功！" << endl;
		showOnePerson(addressBook, result);
	}

	system("pause");
	system("cls");
}

//修改联系人函数
void modifyPerson(AddressBook* addressBook)
{
	cout << "请输入您要修改的联系人：" << endl;
	string name;
	cin >> name;
	int result = isExist(addressBook, name);
	if (result == -1)
	{
		cout << "查无此人！" << endl;
	}
	else
	{
		string name;
		cout << "请输入姓名：" << endl;
		cin >> name;
		addressBook->personArray[result].name = name;

		cout << "请输入性别：" << endl;
		cout << "1--男 2--女" << endl;
		int sex = 0;
		while (true)
		{
			cin >> sex;
			if (sex == 1 || sex == 2)
			{
				addressBook->personArray[result].sex = sex;
				break;
			}
			cout << "输入有误！请重新输入！" << endl;
		}

		cout << "请输入年龄：" << endl;
		int age = 0;
		while (true)
		{
			cin >> age;
			if (age >= 0 && age <= 100)
			{
				addressBook->personArray[result].age = age;
				break;
			}
			cout << "输入有误！请重新输入！" << endl;
		}

		cout << "请输入联系电话：" << endl;
		string phone;
		cin >> phone;
		addressBook->personArray[result].phone = phone;

		cout << "请输入家庭地址：" << endl;
		string address;
		cin >> address;
		addressBook->personArray[result].addr = address;
	}

	system("pause");
	system("cls");
}

//清空所有联系人函数
void cleanPerson(AddressBook* addressBook)
{
	//只需将通讯录的联系人数量置为0，逻辑清空
	addressBook->len = 0;
	cout << "通讯录已清空！" << endl;
	system("pause");
	system("cls");
}

int main()
{
	//创建通讯录结构体变量
	AddressBook addressBook;
	//初始化通讯录人员个数
	addressBook.len = 0;

	int select = 0; // 创建用户选择输入的变量

	while (true)
	{
		// 菜单调用
		showMenu();
		cin >> select;
		switch (select)
		{
		case 1: // 添加联系人
			addPerson(&addressBook);
			break;
		case 2: //显示联系人
			showPerson(&addressBook);
			break;
		case 3: //删除联系人
			deletePerson(&addressBook);
			break;
		case 4: //查找联系人
			findPerson(&addressBook);
			break;
		case 5: //修改联系人
			modifyPerson(&addressBook);
			break;
		case 6: //清空联系人
			cleanPerson(&addressBook);
			break;
		case 0: // 退出通讯录
			cout << "欢迎下次使用" << endl;
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