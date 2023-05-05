#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Заполнение массива случайными числами от 0 до 249
void fill_array(int* arr, int length) {
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) {
        arr[i] = rand() % 250;
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


__global__ void bitonic_Sort_Step(int* deviceArr, int j, int k, int length) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x * 2; //произведение размерности блока и номера блока, 
    //к которому принадлежит данная нить, умноженное на 2, а затем увеличенное на индекс нити в блоке.
    
    ixj = i ^ j;

    if ((i < length && ixj < length) && (ixj) > i) {  //  ixj больше значения i для того, 
                                                      //чтобы избежать повторной обработки элементов в других блоках. 
        if ((i & k) == 0) { //i четная
            /* Сортировка по возрастанию */
            if (deviceArr[i] > deviceArr[ixj]) {
                int temp = deviceArr[i];
                deviceArr[i] = deviceArr[ixj];
                deviceArr[ixj] = temp;
            }
        }
        if ((i & k) != 0) { // i нечетная
            /* Сортировка по возрастанию */
            if (deviceArr[i] > deviceArr[ixj]) {
                int temp = deviceArr[i];
                deviceArr[i] = deviceArr[ixj];
                deviceArr[ixj] = temp;
            }
        }
    }
}

// Алгоритм параллельной сортировки Бэтчера на CPU
void cpuBatcherSort(int* arr, int n) {
    // Рекурсивное разбиение массива на пары элементов и их последующая сортировка
    for (int k = 2; k <= n; k <<= 1) {          // перебираем размеры подмассивов, начиная с 2 и удваивая на каждой итерации
        for (int j = k >> 1; j > 0; j >>= 1) {  // перебираем расстояния между сравниваемыми элементами, начиная с половины размера подмассива и уменьшая вдвое на каждой итерации
            for (int i = 0; i < n; ++i) {       // перебираем элементы в массиве
                int ij = i ^ j;                 // вычисляем индекс соседнего элемента в подмассиве
                if (ij > i) {                   // если найден соседний элемент с большим номером в подмассиве
                    if ((i & k) == 0 && arr[i] > arr[ij]) { // проверяем, нужно ли поменять местами элементы
                        swap(arr[i], arr[ij]);  
                    }
                    if((i & k) != 0 && arr[i] < arr[ij]) {
                        swap(arr[i], arr[ij]);
                    }
                }
            }
        }
    }
}

// Алгоритм параллельной сортировки Бэтчера на GPU
void gpuBatcherSort(int* hostArr, int length) {
    // Выделение памяти на GPU и копирование данных из хоста на GPU
    int* deviceArr = nullptr;
    cudaMalloc((void**)&deviceArr, length * sizeof(int));
    cudaMemcpy(deviceArr, hostArr, length * sizeof(int), cudaMemcpyHostToDevice);

    // Определение размеров блоков и сетки для запуска ядра сортировки Бэтчера
    dim3 block_size(256, 1, 1);
    dim3 grid_size((length / 2 + block_size.x - 1) / block_size.x, 1);

    // Параллельная сортировка Бэтчера на GPU
    for (int k = 2; k <= length; k <<= 1) {          
        for (int j = k >> 1; j > 0; j >>= 1) {  
            bitonic_Sort_Step << <grid_size, block_size >> > (deviceArr, j, k, length);
        }
    }

    cudaDeviceSynchronize();

    // Объединение отсортированных подмассивов в один отсортированный массив
    thrust::device_vector<int> dev_vec_A(deviceArr, deviceArr + length);
    thrust::sort(dev_vec_A.begin(), dev_vec_A.end());
    thrust::copy(dev_vec_A.begin(), dev_vec_A.end(), hostArr);

    // Освобождение памяти на GPU
    cudaFree(deviceArr);
}

int main() {
    const int length = 33554432;
    int* arr_cpu = new int[length];
    int* arr_gpu = new int[length];
    fill_array(arr_cpu, length);
    memcpy(arr_gpu, arr_cpu, length * sizeof(int));

    // cout << "Initial CPU array: ";
    // print_array(arr_cpu, length);

    // cout << "Initial GPU array: ";
    // print_array(arr_gpu, length);

    // Запуск сортировки на CPU
    auto start_cpu = high_resolution_clock::now();
    cpuBatcherSort(arr_cpu, length);
    auto stop_cpu = high_resolution_clock::now();

    // Запуск сортировки на GPU
    auto start_gpu = high_resolution_clock::now();
    gpuBatcherSort(arr_gpu, length);
    auto stop_gpu = high_resolution_clock::now();

    // cout << "Sorted array CPU : ";
    // print_array(arr_cpu, length);

    // cout << "Sorted array GPU: ";
    // print_array(arr_gpu, length);

    // Проверка корректности сортировки
    if (comparison_arrays(arr_cpu, arr_gpu, length)) {
        cout << "Sorting was successful" << endl;
    }
    else {
        cout << "Sorting failed" << endl;
    }

    for (int i = 0; i < length; ++i) {
        if (arr_cpu[i] != arr_gpu[i]) {
            printf("Sorting failed at index %d\n", i);
            break;
        }
    }

    // Вывод времени выполнения сортировки на CPU и GPU
    auto duration_cpu = duration_cast<microseconds>(stop_cpu - start_cpu);
    auto duration_gpu = duration_cast<microseconds>(stop_gpu - start_gpu);
    cout << "Execution time on CPU: " << duration_cpu.count() << " mcs" << endl;
    cout << "Execution time on GPU: " << duration_gpu.count() << " mcs" << endl;

    delete[] arr_cpu;
    delete[] arr_gpu;

    return 0;
}