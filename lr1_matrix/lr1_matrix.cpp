#include <iostream>
#include <Windows.h>
#include <random>
#include <chrono>
#include <vector>
#include <omp.h>
using namespace std;
using clk = std::chrono::steady_clock;


// Функция генерации матриц n*m размерности
vector<vector<double>> randomMatrix(size_t n, size_t m, unsigned int seed = 12345, double minVal = -1000.0, double maxVal = 1000.0) {
    mt19937 gen(seed); // генератор Marsenne Twister
    cout << "Генератор матрицы (seed) = " << seed << endl;
    uniform_real_distribution<> dist(minVal, maxVal); // равномерное распределение

    vector<vector<double>> matrix(n, vector<double>(m, 0.0)); // n строк, m столбцов, инициализация нулями

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            matrix[i][j] = dist(gen);
        }
    }

    return matrix;
}

vector<double> randomMatrixFlat(size_t n, size_t m,
    unsigned int seed = 12345,
    double minVal = -1000.0, double maxVal = 1000.0)
{
    mt19937 gen(seed);
    cout << "Генератор матрицы (seed) = " << seed << endl;

    uniform_real_distribution<> dist(minVal, maxVal); // равномерное распределение

    vector<double> matrix(n * m);

    for (size_t i = 0; i < n * m; ++i)
        matrix[i] = dist(gen);

    return matrix;
}

inline double& at(vector<double>& mat, size_t ncols, size_t i, size_t j) {
    return mat[i * ncols + j];
}

pair<size_t, size_t> getMatrixSize(const vector<vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix.empty() ? 0 : matrix[0].size();
    return { rows, cols };
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


void measureTimeSeq(vector<double>& matrix, int numMeasurements) {
    double seqTotalTime = 0;
    //double result;

    double maxv = -numeric_limits<double>::infinity();
    // Многократные замеры для последовательного алгоритма
    for (int i = 0; i < numMeasurements; ++i) {
        auto t0 = clk::now(); // старт измерения времени выполнения
        //result = getMaxMatrix(matrix);
        for (size_t i = 0; i < matrix.size(); ++i)
            maxv = maxv > matrix[i] ? maxv : matrix[i];

        auto t1 = clk::now(); // окончание измерения времени

        static volatile double sink; // защищаем от выкидывания
        //sink = result;
        sink - maxv;

        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        seqTotalTime += dt;
    }

    double seqAvgTime = seqTotalTime / numMeasurements;
    cout << " | Последовательный: " << seqAvgTime << " мкс; " << "max=" << maxv << endl;
}

void measureTimePar(vector<double>& matrix, int numMeasurements) {
    double parTotalTime = 0;
    double t0 = 0.0; // ОБЩЕЕ время для всех потоков
    double maxv = -numeric_limits<double>::infinity();
    //pair<size_t, size_t> size = getMatrixSize(matrix);

    #pragma omp parallel
    {
        // Многократные замеры для параллельного алгоритма
        for (int i = 0; i < numMeasurements; ++i) {
            #pragma omp barrier

            #pragma omp single
            {
                t0 = omp_get_wtime();
            }

            double local_max = maxv;
            /*
            #pragma omp for
            for (int i = 0; i < size.first; i++) {
                for (int j = 0; j < size.second; j++) {
                    if (matrix[i][j] > maxv) {
                        local_max = matrix[i][j];
                    }
                }
            }
            */

            for (int i = 0; i < matrix.size(); i++) {
                local_max = local_max > matrix[i] ? maxv : matrix[i];
            }

            #pragma omp critical
            if (local_max > maxv) maxv = local_max;

            #pragma omp single
            {
                double dt = (omp_get_wtime() - t0) * 1e6; // умножаем для микросекунд
                parTotalTime += dt;
                static volatile double sink; sink = maxv; // не даём выкинуть
            }
        }
    }

    double parAvgTime = parTotalTime / numMeasurements;
    cout << " | Параллельный OpenMP: " << parAvgTime << " мкс; " << "max=" << maxv << endl;
}

void measureTimeParOpt(vector<double>& matrix, int numMeasurements) {
    double parTotalTimeOpt = 0;
    double t0 = 0.0; // ОБЩЕЕ время для всех потоков
    double maxv = -numeric_limits<double>::infinity();
    //pair<size_t, size_t> size = getMatrixSize(matrix);


    #pragma omp parallel
    {
        // Многократные замеры для параллельного алгоритма
        for (int i = 0; i < numMeasurements; ++i) {
            #pragma omp barrier

            #pragma omp single
            {
                t0 = omp_get_wtime();
            }

            /*
            #pragma omp for reduction(max:maxv) schedule(static)
            for (int i = 0; i < size.first; i++) {
                #pragma omp simd reduction(max:maxv)
                for (int j = 0; j < size.second; j++) {
                    if (matrix[i][j] > maxv) {
                        maxv = matrix[i][j];
                    }
                }
            }
            */
            #pragma omp parallel for reduction(max:maxv) schedule(static)
            for (int i = 0; i < matrix.size(); i++) {
                maxv = maxv > matrix[i] ? maxv : matrix[i];
            }
            #pragma omp single
            {
                double dt = (omp_get_wtime() - t0) * 1e6; // умножаем для микросекунд
                parTotalTimeOpt += dt;
                static volatile double sink; sink = maxv; // не даём выкинуть
            }
        }
    }
    
    double parAvgTimeOpt = parTotalTimeOpt / numMeasurements;
    cout << " | Параллельный OpenMP оптимизированный: " << parAvgTimeOpt << " мкс; " << "max=" << maxv << endl;
}

int main()
{
    SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода

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

    //cout << "Размерность матрицы: " << n << " x " << m << endl;
    random_device rd;
    //vector<vector<double>> matrix = randomMatrix(n, m, rd());
    vector<double> matrixFlat = randomMatrixFlat(n, m, rd());
    cout <<endl << "Результаты " << numMeasurements << " прогонов" << " по алгоритмам:" << endl;
    measureTimeSeq(matrixFlat, numMeasurements);
    measureTimePar(matrixFlat, numMeasurements);
    measureTimeParOpt(matrixFlat, numMeasurements);
    return 0;

}
