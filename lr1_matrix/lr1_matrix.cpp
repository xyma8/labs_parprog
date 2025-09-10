#include <iostream>
#include <Windows.h>
#include <random>
#include <chrono>
#include <vector>
#include <omp.h>
using namespace std;
using clk = std::chrono::steady_clock;


// Функция генерации матриц n*m размерности
vector<vector<double>> randomMatrix(size_t n, size_t m, double minVal = -1000.0, double maxVal = 1000.0) {
    //random_device rd;
    mt19937 gen(12345); // генератор Marsenne Twister
    uniform_real_distribution<> dist(minVal, maxVal); //

    vector<vector<double>> matrix(n, vector<double>(m, 0.0)); // n строк, m столбцов, инициализация нулями

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

// Функция нахождения максимального элемента в матрице
int getMaxMatrix(vector<vector<double>> matrix) {
    int max = matrix[0][0];

    int n = matrix.size();
    int m = matrix[0].size();

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            if (matrix[i][j] > max) {
                max = matrix[i][j];
            }
        }
    }

    return max;
}

void printMatrix(vector<vector<int>> matrix) {
    int n = matrix.size();
    int m = matrix[0].size();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << matrix[i][j] << "\t"; // \t = табуляция для выравнивания
        }
        cout << endl;
    }
}

void printMatrix(vector<vector<double>> matrix) {
    int n = matrix.size();
    int m = matrix[0].size();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << matrix[i][j] << "\t"; // \t = табуляция для выравнивания
        }
        cout << endl;
    }
}

void printExecutionTime(clk::time_point t0, clk::time_point t1) {
    // вычисление времени выполнения (миллисекунды)
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << ms << " мс" << endl;

    // вычисление времени выполнения (микроскунды)
    auto mcs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    cout << mcs << " мкс" << endl;
}

int main()
{
    SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода
    
    int temp;
    cout << "Введите количество строк: ";
    if (!(cin >> temp) || temp <= 0) {
        cerr << "Ошибка: кол-во строк должно быть положительным числом" << endl;
        return 1;
    }
    size_t n = static_cast<size_t>(temp);

    cout << "Введите количество столбцов: ";
    if (!(cin >> temp) || temp <= 0) {
        cerr << "Ошибка: кол-во столбцов должно быть положительным числом" << endl;
        return 1;
    }
    size_t m = static_cast<size_t>(temp);

    vector<vector<double>> matrix = randomMatrix(n, m);
    printMatrix(matrix);

    clk::time_point t0 = clk::now(); // старт измерения времени выполнения

    double result = getMaxMatrix(matrix);

    clk::time_point t1 = clk::now(); // окончание измерения времени выполнения

    cout << "Максимальный элемент матрицы = " << result << endl;
    cout << endl;
    cout << "Время выполнения функции нахождения макс. элемента: " << endl;
    printExecutionTime(t0, t1);

    return 0;
}
