#include <iostream>
using namespace std;
#include <string>



class Student
{
public:  // 公共权限
	string name; // 姓名
	int id; // 学号

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
		cout << "姓名： " << name 
			 << "\t学号： " << id << endl;
	}

};
int main()
{
	Student s1 = new Student();
	s1.setID(1541546);
	s1.setName("张三");
	s1.showStudent();

	system("pause");
	return 0;
}