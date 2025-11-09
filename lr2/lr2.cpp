#include <iostream>
//#include <Windows.h>
#include <random>
#include <chrono>
#include <vector>
#include <omp.h>
using namespace std;
using clk = std::chrono::steady_clock;


// Функция генерации матриц n*m размерности
vector<vector<double>> randomMatrix(size_t n, size_t m, unsigned int seed = 12345, double minVal = -10000.0, double maxVal = 10000.0) {
    mt19937 gen(seed); // генератор Marsenne Twister
    cout << "Генератор матрицы (seed) = " << seed << endl;
    normal_distribution<> dist(minVal, maxVal);

    vector<vector<double>> matrix(n, vector<double>(m, 0.0)); // n строк, m столбцов, инициализация нулями

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

// вспомогательная функция нахождения размера матрицы
pair<size_t, size_t> getMatrixSize(const vector<vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix.empty() ? 0 : matrix[0].size();
    return { rows, cols };
}

// Метод подсчета времени выполнения последовательного выполнения
void measureTimeSeq(vector<vector<double>>& matrix, int numMeasurements) {
    double seqTotalTime = 0;
    int n = matrix.size();
    int m = matrix[0].size();
    double sum = 0.0;

    // Многократные замеры для последовательного алгоритма
    for (int i = 0; i < numMeasurements; ++i) {
        sum = 0.0; // сбрасываем перед каждым прогоном, чтобы заново искалось
        auto t0 = clk::now(); // старт измерения времени выполнения

        for (const auto& row : matrix) {
          for (double x : row) {
            sum += x;
          }
        }

        auto t1 = clk::now(); // окончание измерения времени

        static volatile double sink; // защищаем от выкидывания
        sink = sum;

        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        seqTotalTime += dt;
    }

    double seqAvgTime = seqTotalTime / numMeasurements;
    cout << " | Последовательный: " << seqAvgTime << " мкс; " << "sum=" << sum << endl;
}

// Метод подсчета времени выполнения параллельного выполнения
void measureTimePar(vector<vector<double>>& matrix, int numMeasurements) {
    double parTotalTime = 0;
    double sum = 0.0;
    pair<size_t, size_t> size = getMatrixSize(matrix);

    // Многократные замеры для параллельного алгоритма
    for (int i = 0; i < numMeasurements; ++i) {
        sum = 0.0;
        double t0 = omp_get_wtime();

        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                sum += matrix[i][j];
            }
        }

        double dt = (omp_get_wtime() - t0) * 1e6; // умножаем для микросекунд
        parTotalTime += dt;
        static volatile double sink; sink = sum; // не даём выкинуть
    }

    double parAvgTime = parTotalTime / numMeasurements;
    cout << " | Параллельный OpenMP: " << parAvgTime << " мкс; " << "sum=" << sum << endl;
}

// Метод подсчета времени выполнения параллельного с оптимизацией выполнения
void measureTimeParOpt(vector<vector<double>>& matrix, int numMeasurements) {
    double parTotalTimeOpt = 0;
    double sum = 0.0;
    pair<size_t, size_t> size = getMatrixSize(matrix);

    // Многократные замеры для параллельного алгоритма
    for (int i = 0; i < numMeasurements; ++i) {
        sum = 0.0;
        double t0 = omp_get_wtime();

        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (size_t i = 0; i < matrix.size(); ++i) {
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                sum += matrix[i][j];
            }
        }

        double dt = (omp_get_wtime() - t0) * 1e6; // умножаем для микросекунд
        parTotalTimeOpt += dt;
        static volatile double sink; sink = sum; // не даём выкинуть
    }
    
    double parAvgTimeOpt = parTotalTimeOpt / numMeasurements;
    cout << " | Параллельный OpenMP оптимизированный: " << parAvgTimeOpt << " мкс; " << "max=" << sum << endl;
}

int main()
{
    //SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    //SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода

    // фиксируем число потоков, чтобы измерения были стабильны
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_max_threads());

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

    random_device rd;
    vector<vector<double>> matrix = randomMatrix(n, m, rd());

    cout <<endl << "Результаты " << numMeasurements << " прогонов" << " по алгоритмам:" << endl;
    measureTimeSeq(matrix, numMeasurements);
    measureTimePar(matrix, numMeasurements);
    measureTimeParOpt(matrix, numMeasurements);

    return 0;

}
