#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>

// Макрос для максимального количества потоков в блоке
#define MAX_THREADS_PER_BLOCK 1024

using namespace std;
using namespace std::chrono;

// Заполнение массива
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

// Создание копии массива
int* copy_array(int* sourse, int length) {
    int* dest = new int[length];
    for (int i = 0; i < length; i++) {
        dest[i] = sourse[i];
    }
    return dest;
}


// Шаг сортировки Битоническим алгоритмом на GPU
__global__ void bitonic_Sort_Step(int* deviceArr, int j, int k) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x; 
    ixj = i ^ j;

    if ((ixj) > i) {    
        if ((i & k) == 0) { // Сортировка по возрастанию
            if (deviceArr[i] > deviceArr[ixj]) { // Если элементы в неправильном порядке, меняем их местами
                int temp = deviceArr[i];
                deviceArr[i] = deviceArr[ixj];
                deviceArr[ixj] = temp;
            }
        }
        if ((i & k) != 0) { // Сортировка по убыванию
            if (deviceArr[i] < deviceArr[ixj]) { // Если элементы в неправильном порядке, меняем их местами
                int temp = deviceArr[i];
                deviceArr[i] = deviceArr[ixj];
                deviceArr[ixj] = temp;
            }
        }
    }
}

// Алгоритм параллельной сортировки Бэтчера на GPU
void bitonic_Sort(int* arr, int length) {
    int* deviceArr; // Указатель на массив в памяти GPU
    size_t size = length * sizeof(int); // Размер массива

    cudaMalloc((void**)&deviceArr, size); // Выделение памяти на GPU
    cudaMemcpy(deviceArr, arr, size, cudaMemcpyHostToDevice);//Копирование данных из массива CPU в массив GPU

    int threads = MAX_THREADS_PER_BLOCK > length ? length : MAX_THREADS_PER_BLOCK; // Количества потоков
    int blocks = length / threads; //Количество блоков

    // Внешний цикл увеличивает длину подмассива, внутренний проходит по всем элементам подмассива и сравнивает пары элементов 
    for (int k = 2; k <= length; k = k << 1) { 
        for (int j = k >> 1; j > 0; j = j >> 1) {
            bitonic_Sort_Step <<<blocks, threads>>> (deviceArr, j, k); 

            // // Выводим отсортированный массив после каждой итерации
            // cudaMemcpy(arr, deviceArr, size, cudaMemcpyDeviceToHost);

            // printf("After iteration with j=%d and k=%d:\n", j, k);
            // for (int m = 0; m < length; m++) {
            //     printf("%d ", arr[m]);
            // }
            // printf("\n");
        }
    }

    cudaMemcpy(arr, deviceArr, size, cudaMemcpyDeviceToHost); // Копирование сортированного массива обратно на CPU 
    cudaFree(deviceArr); //Освобождение памяти на GPU
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


int main(int argc, char** argv) {
    const int length = 33554432;
    // Создание и заполнение массива
    int* arr_cpu = new int[length];
    int* arr_gpu = new int[length];

    fill_array(arr_cpu, length);
    memcpy(arr_gpu, arr_cpu, length * sizeof(int));


    // cout << "Initial CPU array: ";
    // print_array(arr_cpu, length);

    // cout << endl;

    // cout << "Initial GPU array: ";
    // print_array(arr_gpu, length);
    // cout << endl;


    // Сортировка битоническим алгоритмом на GPU
    // Запуск сортировки на GPU
    auto start_gpu = high_resolution_clock::now();
    bitonic_Sort(arr_gpu, length);
    auto stop_gpu = high_resolution_clock::now();

    // Запуск сортировки на CPU
    auto start_cpu = high_resolution_clock::now();
    cpuBatcherSort(arr_cpu, length);
    auto stop_cpu = high_resolution_clock::now();

    // cout << endl;
    // cout << "Sorted array GPU: ";
    // print_array(arr_gpu, length);

    // cout << endl;

    // cout << "Sorted array CPU : ";
    // print_array(arr_cpu, length);

    // cout << endl;


    bool isSortedCorrectly = comparison_arrays(arr_gpu, arr_cpu, length);

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
    auto duration_cpu = duration_cast<milliseconds>(stop_cpu - start_cpu);
    auto duration_gpu = duration_cast<milliseconds>(stop_gpu - start_gpu);
    cout << "Execution time on CPU: " << duration_cpu.count() << " ms" << endl;
    cout << "Execution time on GPU: " << duration_gpu.count() << " ms" << endl;

    delete[] arr_gpu;
    delete[] arr_cpu;

    return 0;
}