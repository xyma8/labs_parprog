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
    normal_distribution<> dist(minVal, maxVal); //

    vector<vector<double>> matrix(n, vector<double>(m, 0.0)); // n строк, m столбцов, инициализация нулями

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

// Функция нахождения максимального элемента в матрице
double getMaxMatrix(vector<vector<double>>& matrix) {
    double maxv = -numeric_limits<double>::infinity();

    int n = matrix.size();
    int m = matrix[0].size();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (matrix[i][j] > maxv) {
                maxv = matrix[i][j];
            }
        }
    }

    return maxv;
}


double measureExecutionTime(int n, int m) {
    vector<vector<double>> matrix = randomMatrix(n, m);

    auto t0 = clk::now(); // старт измерения времени выполнения
    double result = getMaxMatrix(matrix);
    auto t1 = clk::now(); // окончание измерения времени

    static volatile double sink; // защищаем от выкидывания
    sink = result;

    // возвращаем время в микросекундах (мкс)
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

void measureTimeForSizes(int n, int m, int numMeasurements) {
    double seqTotalTime = 0;
    double parTotalTime = 0;

    // Многократные замеры для последовательного алгоритма
    for (int i = 0; i < numMeasurements; ++i) {
        seqTotalTime += measureExecutionTime(n, m);
    }

    vector<vector<double>> matrix;
    double t0 = 0.0; // ОБЩЕЕ для всех потоков (время)
    double maxv = -numeric_limits<double>::infinity();

    #pragma omp parallel
    {
        // Многократные замеры для параллельного алгоритма
        for (int i = 0; i < numMeasurements; ++i) {
            #pragma omp barrier

            #pragma omp single
            {
                matrix = randomMatrix(n, m);
                t0 = omp_get_wtime();
            }
            #pragma omp barrier

            #pragma omp for
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    if (matrix[i][j] > maxv) {
                        maxv = matrix[i][j];
                    }
                }
            }

            #pragma omp barrier
            #pragma omp single
            {
                double dt = (omp_get_wtime() - t0) * 1e6; // умножаем для микросекунд
                parTotalTime += dt;
                static volatile double sink; sink = maxv; // не даём выкинуть
            }
        }
    }

    // Среднее время для каждого из вариантов
    double seqAvgTime = seqTotalTime / numMeasurements;
    double parAvgTime = parTotalTime / numMeasurements;

    // Вывод результатов
    cout << "Размерность: " << n <<" x "<<m
        << " | Последовательный: " << seqAvgTime << " мкс"
        << " | Параллельный OpenMP: " << parAvgTime << " мкс\n";
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

    cout << "Введите количество прогонов: ";
    if (!(cin >> temp) || temp <= 0) {
        cerr << "Ошибка: кол-во прогонов должно быть положительным числом" << endl;
        return 1;
    }
    size_t numMeasurements = static_cast<size_t>(temp);

    measureTimeForSizes(n,m, numMeasurements);

    
    return 0;

}
