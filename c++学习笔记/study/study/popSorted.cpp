#include <iostream>
using namespace std;

int popSorted() {

	int arr[7] = { 1,3,4,5,6,8,2 };
	int end = sizeof(arr) / sizeof(arr[0]);
	// Ã°ÅÝÅÅÐò
	for (int i = end - 1; i > 0; i--) {
		for (int j = 0; j < i; j++)
		{
			if (arr[j] > arr[j + 1]) {
				int temp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = temp;
			}
		}
	}

	for (int i = 0; i < end; i++) {
		cout << arr[i] << endl;
	}

	system("pause");
	return 0;
}