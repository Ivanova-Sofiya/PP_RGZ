#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <algorithm>

#define MAX_THREADS_PER_BLOCK 1024
using namespace std;

// Заполнение массива
void fill_array(int* arr, int length) {
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) {
        arr[i] = rand();
    }
}

// Дополнение массива доп элементами
void additional_elements(int* arr, int sourseLength, int destLength) {
    int maxValue = *max_element(arr, arr + sourseLength);

    for (int i = sourseLength; i < destLength; i++) {
        arr[i] = maxValue + 1;
    }
}

// Сравнение массивов
bool comparison_arrays(int* arr1, int* arr2, int length) {
    for (int i = 0; i < length; i++) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

// Вывод массива
void print_array(int* arr, int length) {
    int i;
    for (i = 0; i < length; ++i) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Массив копий
int* copy_array(int* sourse, int length) {
    int* dest = new int[length];

    for (int i = 0; i < length; i++) {
        dest[i] = sourse[i];
    }
    return dest;
}

//Ближайшая степень двойки, которая больше или равна заданному числу
int powerCeil(int x) {
    if (x <= 1) return 1;
    int power = 2;
    x--;
    while (x >>= 1) power <<= 1;
    return power;
}

__global__ void bitonic_Sort_Step(int* deviceArr, int j, int k) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x; 
    ixj = i ^ j;

    if ((ixj) > i) {    
        if ((i & k) == 0) {
            /* Сортировка по возрастанию */
            if (deviceArr[i] > deviceArr[ixj]) {
                /* exchange(i,ixj); */
                int temp = deviceArr[i];
                deviceArr[i] = deviceArr[ixj];
                deviceArr[ixj] = temp;
            }
        }
        if ((i & k) != 0) {
            /* Сортировка по убыванию */
            if (deviceArr[i] < deviceArr[ixj]) {
                /* exchange(i,ixj); */
                int temp = deviceArr[i];
                deviceArr[i] = deviceArr[ixj];
                deviceArr[ixj] = temp;
            }
        }
    }
}

void bitonic_Sort(int* arr, int length) {
    int* deviceArr;
    size_t size = length * sizeof(int);

    cudaMalloc((void**)&deviceArr, size); //выделяем память
    cudaMemcpy(deviceArr, arr, size, cudaMemcpyHostToDevice);//копирование 

    int threads = MAX_THREADS_PER_BLOCK > length ? length : MAX_THREADS_PER_BLOCK; //Количество потоков
    int blocks = length / threads; //Количество блоков

    for (int k = 2; k <= length; k = k << 1) { 
        for (int j = k >> 1; j > 0; j = j >> 1) {
            bitonic_Sort_Step <<<blocks, threads>>> (deviceArr, j, k); 
        }
    }
    cudaMemcpy(arr, deviceArr, size, cudaMemcpyDeviceToHost); //копия сортированного массива 
    cudaFree(deviceArr); //освобождение памяти
}

//Проверяем является ли массив битоническим 
bool isBitonic(int*v, int length) {
    bool wasDecreasing = v[length - 1] > v[0]; //проверяем был ли предыдущий элемент меньше текущего
    int numInflections = 0;
    for (int i = 0; i < length && numInflections <= 2; i++) {
        bool isDecreasing = v[i] > v[(i + 1) % length];
        // Check if this element and next one are an inflection.
        if (wasDecreasing != isDecreasing) {
            numInflections++;
            wasDecreasing = isDecreasing;
        }
    }
    return 2 == numInflections; // Если точек перегиба две, то массив считается битоническим 
}

int main(void) {
    int length = 0;
    cout << "Specify the length of the array: ";
    cin >> length;

    int roundingLength = powerCeil(length);
    int* cudaArr = new int[roundingLength];
    fill_array(cudaArr, length);
    additional_elements(cudaArr, length, roundingLength);

// Вывод результатов выполнения программы
    cout << "" << endl;
    cout << "Std sort on CPU ..." << endl;

    clock_t start, stop;
    start = clock();
    int* stdArr = copy_array(cudaArr, length);
    sort(stdArr, stdArr + length);
    stop = clock();
    double elapcedTime = (double)(stop - start) / CLOCKS_PER_SEC;
    cout << "Time for CPU: " << elapcedTime << " sec" << endl << endl;

    cout << "Bitonic sort on GPU..." << endl;
    start = clock();
    bitonic_Sort(cudaArr, roundingLength);
    stop = clock();
    elapcedTime = (double)(stop - start) / CLOCKS_PER_SEC;
    cout << "Time for GPU: " << elapcedTime << " sec" << endl << endl;
}
