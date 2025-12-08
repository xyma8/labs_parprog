#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>

using namespace std;
using clk = chrono::steady_clock;

// Генерация матрицы n×m размерности
vector<vector<double>> randomMatrix(size_t n, size_t m, unsigned int seed = 12345, double minVal = -10000.0, double maxVal = 10000.0) {
    mt19937 gen(seed);
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
    return {rows, cols};
}

// 1. Последовательный алгоритм
void measureTimeSeq(vector<vector<double>>& matrix, int numMeasurements) {
    double seqTotalTime = 0;
    double maxv = -numeric_limits<double>::infinity();
    int n = matrix.size();
    int m = matrix[0].size();

    for (int k = 0; k < numMeasurements; ++k) {
        auto t0 = clk::now();
        maxv = -numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                if (matrix[i][j] > maxv)
                    maxv = matrix[i][j];
        auto t1 = clk::now();
        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        seqTotalTime += dt;
    }

    double avg = seqTotalTime / numMeasurements;
    cout << " | Последовательный: " << avg << " мкс; " << "max=" << maxv << endl;
}

// 2. Параллельный MPI (без оптимизации)
void measureTimePar(vector<vector<double>>& matrix, int numMeasurements, int rank, int size) {
    pair<size_t, size_t> dim = getMatrixSize(matrix);
    double parTotalTime = 0;
    double globalMax = -numeric_limits<double>::infinity();

    // Делим строки между процессами
    int rowsPerProc = dim.first / size;
    int extra = dim.first % size;
    int startRow = rank * rowsPerProc + min(rank, extra);
    int endRow = startRow + rowsPerProc + (rank < extra ? 1 : 0);

    vector<vector<double>> localPart(endRow - startRow);
    for (int i = 0; i < endRow - startRow; ++i)
        localPart[i] = matrix[startRow + i];

    for (int k = 0; k < numMeasurements; ++k) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // Локальный максимум
        double localMax = -numeric_limits<double>::infinity();
        for (auto& row : localPart)
            for (double val : row)
                if (val > localMax) localMax = val;

        // Обмен максимумами (точка-точка)
        if (rank == 0) {
            globalMax = localMax;
            for (int src = 1; src < size; ++src) {
                double recvVal;
                MPI_Recv(&recvVal, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (recvVal > globalMax) globalMax = recvVal;
            }
        } else {
            MPI_Send(&localMax, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        double dt = (t1 - t0) * 1e6;

        // Суммируем только на главном процессе
        if (rank == 0) parTotalTime += dt;
    }

    if (rank == 0) {
        double avg = parTotalTime / numMeasurements;
        cout << " | Параллельный MPI: " << avg << " мкс; " << "max=" << globalMax << endl;
    }
}

// Параллельный MPI с оптимизацией
void measureTimeParOpt(vector<vector<double>>& matrix, int numMeasurements, int rank, int size) {
    pair<size_t, size_t> dim = getMatrixSize(matrix);
    double parTotalTimeOpt = 0;
    double globalMax = -numeric_limits<double>::infinity();

    int rowsPerProc = dim.first / size;
    int extra = dim.first % size;
    int startRow = rank * rowsPerProc + min(rank, extra);
    int endRow = startRow + rowsPerProc + (rank < extra ? 1 : 0);

    vector<vector<double>> localPart(endRow - startRow);
    for (int i = 0; i < endRow - startRow; ++i)
        localPart[i] = matrix[startRow + i];

    for (int k = 0; k < numMeasurements; ++k) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // локальный максимум с векторизацией
        double localMax = -numeric_limits<double>::infinity();
        for (auto& row : localPart) {
            // Оптимизация: использовать обычный цикл (компилятор сам векторизует)
            for (double val : row)
                if (val > localMax) localMax = val;
        }

        // Оптимизированная передача: цепочка передач вместо звезды
        if (rank == 0) {
            globalMax = localMax;
            double recvVal;
            MPI_Send(&globalMax, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&recvVal, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            globalMax = recvVal;
        } else {
            double recvVal;
            MPI_Recv(&recvVal, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (localMax > recvVal) recvVal = localMax;
            MPI_Send(&recvVal, 1, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        double dt = (t1 - t0) * 1e6;

        if (rank == 0) parTotalTimeOpt += dt;
    }

    if (rank == 0) {
        double avg = parTotalTimeOpt / numMeasurements;
        cout << " | Параллельный MPI оптимизированный: " << avg << " мкс; " << "max=" << globalMax << endl;
    }
}

// === main ===
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n, m, numMeasurements;

    if (rank == 0) {
        cout << "Введите количество строк: ";
        cin >> n;
        cout << "Введите количество столбцов: ";
        cin >> m;
        cout << "Введите количество прогонов: ";
        cin >> numMeasurements;
    }

    // Рассылка параметров
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numMeasurements, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    vector<vector<double>> matrix;
    if (rank == 0) {
        random_device rd;
        matrix = randomMatrix(n, m, rd());
    } else {
        matrix = vector<vector<double>>(n, vector<double>(m, 0.0));
    }

    // Рассылаем матрицу всем процессам (каждый получит копию для простоты)
    for (size_t i = 0; i < n; ++i)
        MPI_Bcast(matrix[i].data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << endl << "Результаты " << numMeasurements << " прогонов по алгоритмам:" << endl;
    }

    if (rank == 0) measureTimeSeq(matrix, numMeasurements);
    measureTimePar(matrix, numMeasurements, rank, size);
    measureTimeParOpt(matrix, numMeasurements, rank, size);

    MPI_Finalize();
    return 0;
}
