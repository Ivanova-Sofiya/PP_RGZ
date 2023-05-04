#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
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
                    if ((i & k) != 0 && arr[i] < arr[ij]) { // проверяем, нужно ли поменять местами элементы
                        swap(arr[i], arr[ij]); 
                    }
                }
            }
        }
    }
}

int main() {
    int length = 33554432;      //количество наших элементов
    int* arr_cpu = new int[length];

    //Заполняем массив
    fill_array(arr_cpu, length);
    // cout << "Source array:" << endl;
    // print_array(arr_cpu, length);


    auto start_time_cpu = high_resolution_clock::now();
    // Сортировка массива на CPU с помощью алгоритма Бэтчера + замеры времени 
    cpuBatcherSort(arr_cpu, length);
    auto end_time_cpu = high_resolution_clock::now();

    // Копирование массива на GPU
    thrust::device_vector<int> dev_vec(length); //Создает вектор устройства целых чисел заданной длины на GPU устройстве
    thrust::copy(arr_cpu, arr_cpu + length, dev_vec.begin()); //Копирует содержимое массива целых чисел из памяти CPU в вектор устройства

    dim3 dimBlock(1024, 1, 1); //Выставляет размер блока нитей на 1024 нити вдоль одной оси и по одной нити вдоль двух остальных осей.
    dim3 dimGrid((length + dimBlock.x - 1) / dimBlock.x, 1, 1); // Вычисляет количество блоков нитей, которые должны быть запущены на GPU, 
                                                                //чтобы обработать весь массив данных. 

    auto start_time_gpu = high_resolution_clock::now();

    //Сортировка массива на GPU с использованием CUDA
    for (int len = 2; len <= length; len <<= 1) {
        for (int i = len >> 1; i > 0; i >>= 1) {
            bitonic_Sort_Step<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dev_vec.data()), i, len / 2, length);
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) { //шаблон обработки ошибок
                printf("CUDA error: %s\n", cudaGetErrorString(error));
                exit(-1);
            }
            cudaDeviceSynchronize(); //это функция из библиотеки CUDA Runtime API, которая блокирует хост-программу 
                                     //до тех пор, пока все предыдущие операции, отправленные на устройство CUDA, не завершатся.
        }
    }

    // Копирование отсортированного массива обратно на CPU
    int* arr_gpu = new int[length]; //выделяем память 
    thrust::copy(dev_vec.begin(), dev_vec.end(), arr_gpu); //копируем

    auto end_time_gpu = high_resolution_clock::now();

    bool is_sorted_correctly = comparison_arrays(arr_cpu, arr_gpu, length); //сравниваем получившиеся массивы

    // cout << "Sorted array on CPU:" << endl;
    // print_array(arr_cpu, length);

    // cout << "Sorted array on GPU:" << endl;
    // print_array(arr_gpu, length);
        
    for (int i = 0; i < length; ++i) {
        if (arr_cpu[i] != arr_gpu[i]) {
            printf("Sorting failed at index %d\n", i);
            break;
        }
    }

    auto duration_cpu = duration_cast<microseconds>(end_time_cpu - start_time_cpu);
    auto duration_gpu = duration_cast<microseconds>(end_time_gpu - start_time_gpu);

    cout << "Execution time on CPU: " << duration_cpu.count() << " mcs" << endl;
    cout << "Execution time on GPU: " << duration_gpu.count() << " mcs" << endl;

    //все подчищаем
    delete[] arr_cpu;
    delete[] arr_gpu;

    dev_vec.clear();

    return 0;
}