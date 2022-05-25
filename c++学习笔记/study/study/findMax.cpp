#include <iostream>
using namespace std;

int FindMax() {

	// 找到数组的最大值
	int arr[5] = { 300, 350, 200, 400, 250 };
	int max = 0;
	for (int i = 0; i < 5; i++)
	{
		if (arr[i] > max) max = arr[i];
	}
	// 交换元素
	int temp = arr[0];
	arr[0] = arr[4];
	arr[4] = temp;
	cout << arr[0] << endl;
	cout << max << endl;

	return 0;
}