#pragma once
#include <iostream>
#include <string>

class Worker
{
public:
	Worker();
	~Worker();
	// ��ʾ������Ϣ
	virtual void showInfo() = 0;
	// ��ȡ��λ����
	virtual string getDeptName() = 0;

	int m_Id; //ְ�����
	string m_Name; //ְ������
	int m_DeptId; //ְ�����ڲ������Ʊ��
};

