#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>

using namespace std;
using clk = std::chrono::steady_clock;

// Функция генерации матриц n*m размерности
vector<vector<double>> randomMatrix(size_t n, size_t m, unsigned int seed = 12345, double minVal = -10000.0, double maxVal = 10000.0) {
    mt19937 gen(seed);  // генератор Marsenne Twister
    cout << "Генератор матрицы (seed) = " << seed << endl;
    normal_distribution<> dist(minVal, maxVal);
    vector<vector<double>> matrix(n, vector<double>(m, 0.0)); // n строк, m столбцов, инициализация нулями

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j)
            matrix[i][j] = dist(gen);

    return matrix;
}

// вспомогательная функция нахождения размера матрицы
pair<size_t, size_t> getMatrixSize(const vector<vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix.empty() ? 0 : matrix[0].size();
    return { rows, cols };
}

// 1. Последовательная версия
void measureTimeSeq(vector<vector<double>>& matrix, int numMeasurements) {
    double seqTotalTime = 0;
    int n = matrix.size();
    int m = matrix[0].size();
    double maxv = -numeric_limits<double>::infinity();

    for (int run = 0; run < numMeasurements; ++run) {
        maxv = -numeric_limits<double>::infinity();
        auto t0 = clk::now();

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                if (matrix[i][j] > maxv) maxv = matrix[i][j];

        auto t1 = clk::now();
        double dt = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
        seqTotalTime += dt;
    }

    double seqAvgTime = seqTotalTime / numMeasurements;
    cout << " | Последовательный: " << seqAvgTime << " мкс; max=" << maxv << endl;
}

// 2. Параллельный MPI (без оптимизации)
void measureTimePar(vector<vector<double>>& matrix, int numMeasurements) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double parTotalTime = 0;
    double global_max = -numeric_limits<double>::infinity();
    pair<size_t, size_t> size = getMatrixSize(matrix);
    size_t n = size.first, m = size.second;

    // Разделяем строки между процессами
    size_t rows_per_proc = n / world_size;
    size_t start = world_rank * rows_per_proc;
    size_t end = (world_rank == world_size - 1) ? n : start + rows_per_proc;

    for (int run = 0; run < numMeasurements; ++run) {
        double t0 = MPI_Wtime();

        double local_max = -numeric_limits<double>::infinity();
        for (size_t i = start; i < end; ++i)
            for (size_t j = 0; j < m; ++j)
                if (matrix[i][j] > local_max) local_max = matrix[i][j];

        global_max = 0.0;
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        double dt = (t1 - t0) * 1e6;

        if (world_rank == 0) parTotalTime += dt;
    }

    if (world_rank == 0) {
        double parAvgTime = parTotalTime / numMeasurements;
        cout << " | Параллельный MPI: " << parAvgTime << " мкс;" << "max=" << global_max << endl;
    }
}

// 3. Параллельный MPI с оптимизацией
void measureTimeParOpt(vector<vector<double>>& matrix, int numMeasurements) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double parTotalTimeOpt = 0;
    double global_max = -numeric_limits<double>::infinity();
    pair<size_t, size_t> size = getMatrixSize(matrix);
    size_t n = size.first, m = size.second;

    // Распределение по блокам строк, чтобы нагрузка была равномерная
    size_t rows_per_proc = (n + world_size - 1) / world_size;
    size_t start = world_rank * rows_per_proc;
    size_t end = min(start + rows_per_proc, n);

    // Можно оптимизировать доступ к памяти — заранее выровнять данные
    for (int run = 0; run < numMeasurements; ++run) {
        double t0 = MPI_Wtime();

        // Локальный максимум (векторизованный доступ внутри кэша)
        double local_max = -numeric_limits<double>::infinity();
        for (size_t i = start; i < end; ++i) {
            const auto& row = matrix[i];
            for (double val : row)
                if (val > local_max) local_max = val;
        }

        global_max = 0.0;
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        double dt = (t1 - t0) * 1e6;

        if (world_rank == 0) parTotalTimeOpt += dt;
    }

    if (world_rank == 0) {
        double parAvgTimeOpt = parTotalTimeOpt / numMeasurements;
        cout << " | Параллельный MPI оптимизированный: " << parAvgTimeOpt << " мкс;" << "max=" << global_max << endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int temp;
    size_t n, m, numMeasurements;

    if (world_rank == 0) {
        cout << "Введите количество строк: ";
        if (!(cin >> temp) || temp <= 0) { cerr << "Ошибка: кол-во строк должно быть положительным числом" << endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        n = temp;

        cout << "Введите количество столбцов: ";
        if (!(cin >> temp) || temp <= 0) { cerr << "Ошибка: кол-во столбцов должно быть положительным числом" << endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        m = temp;

        cout << "Введите количество прогонов: ";
        if (!(cin >> temp) || temp <= 0) { cerr << "Ошибка: кол-во прогонов должно быть положительным числом" << endl; MPI_Abort(MPI_COMM_WORLD, 1); }
        numMeasurements = temp;
    }

    // Распространяем параметры всем процессам
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numMeasurements, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    vector<vector<double>> matrix;
    if (world_rank == 0) {
        random_device rd;
        matrix = randomMatrix(n, m, rd());
    } else {
        // Остальные процессы получают пустой контейнер, но размеры известны
        matrix.resize(n, vector<double>(m, 0.0));
    }

    // Распространяем матрицу (простой вариант: broadcast построчно)
    for (size_t i = 0; i < n; ++i)
        MPI_Bcast(matrix[i].data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
        cout << "\nРезультаты " << numMeasurements << " прогонов по алгоритмам:" << endl;

    if (world_rank == 0)
        measureTimeSeq(matrix, numMeasurements);

    measureTimePar(matrix, numMeasurements);
    measureTimeParOpt(matrix, numMeasurements);

    MPI_Finalize();
    return 0;
}
