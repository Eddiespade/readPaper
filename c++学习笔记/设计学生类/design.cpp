#include <iostream>
using namespace std;
#include <string>



class Student
{
public:  // ����Ȩ��
	string name; // ����
	int id; // ѧ��

	void setName(string name)
	{
		this.name = name;
	}

	void setID(int id)
	{
		this.id = id;
	}

	void showStudent()
	{
		cout << "������ " << name 
			 << "\tѧ�ţ� " << id << endl;
	}

};
int main()
{
	Student s1 = new Student();
	s1.setID(1541546);
	s1.setName("����");
	s1.showStudent();

	system("pause");
	return 0;
}