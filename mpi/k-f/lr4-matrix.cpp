#include <mpi.h>
//#include <Windows.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <limits>

using namespace std;
using clk = std::chrono::steady_clock;

// Функция генерации матриц n*m размерности
vector<vector<double>> randomMatrix(size_t n, size_t m, unsigned int seed = 54321, double minVal = -10000.0, double maxVal = 10000.0) {
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

// Последовательное выполнение
void measureTimeSeq(vector<vector<double>>& matrix, int numMeasurements) {
    double seqTotalTime = 0;
    int n = matrix.size();
    int m = matrix[0].size();
    double maxv = -numeric_limits<double>::infinity();

    // Многократные замеры для последовательного алгоритма
    for (int i = 0; i < numMeasurements; ++i) {
        maxv = 0; // сбрасываем перед каждым прогоном, чтобы заново искалось
        auto t0 = clk::now(); // старт измерения времени выполнения

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (matrix[i][j] > maxv) {
                    maxv = matrix[i][j];
                }
            }
        }

        auto t1 = clk::now(); // окончание измерения времени
        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        seqTotalTime += dt;
    }

    double seqAvgTime = seqTotalTime / numMeasurements;
    cout << " | Последовательный: " << seqAvgTime << " мкс; " << "max=" << maxv << endl;
}

// MPI Коллективные функции. Параллельное выполнение
void measureTimePar(vector<vector<double>>& matrix, int numMeasurements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int localN = (int)matrix.size();
    const int m = (localN>0? (int)matrix[0].size() : 0);

    double totalTime = 0.0;
    double globalMax = -numeric_limits<double>::infinity();

    for (int r = 0; r < numMeasurements; ++r) {
        double start = MPI_Wtime();

        // Локальный максимум
        double localMax = -numeric_limits<double>::infinity();
        for (int i = 0; i < localN; ++i)
            for (int j = 0; j < m; ++j)
                if (matrix[i][j] > localMax)
                    localMax = matrix[i][j];

        // Коллективная операция: редукция для нахождения глобального максимума
        MPI_Reduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double end = MPI_Wtime();
        totalTime += (end - start) * 1e6; // микросекунды
    }

    double avgTime = totalTime / numMeasurements;
    if (rank == 0)
        cout << " | Параллельный MPI: "
             << avgTime << " мкс; max=" << globalMax << endl;
}

// Параллельная версия с коллективными функциями MPI
void measureTimeParOpt(const vector<vector<double>>& matrix, int numMeasurements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int localN = (int)matrix.size();
    const int m = (localN>0? (int)matrix[0].size() : 0);

    double totalTime = 0.0;
    double globalMax = -numeric_limits<double>::infinity();

    MPI_Request request;

    for (int r = 0; r < numMeasurements; ++r) {
        double start = MPI_Wtime();

        // Локальный максимум
        double localMax = -numeric_limits<double>::infinity();
        for (int i = 0; i < localN; ++i)
            for (int j = 0; j < m; ++j)
                if (matrix[i][j] > localMax)
                    localMax = matrix[i][j];

        // Асинхронная редукция — не блокирует вычисления
        MPI_Ireduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD, &request);

        // Пока редукция идёт, можно сделать что-то ещё (например, подготовку данных)
        // Для честного измерения просто ждём завершения
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        double end = MPI_Wtime();
        totalTime += (end - start) * 1e6; // микросекунды
    }

    double avgTime = totalTime / numMeasurements;
    if (rank == 0)
        cout << " | Параллельный MPI (оптимизированный): "
             << avgTime << " мкс; max=" << globalMax << endl;
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 0, m = 0, numMeasurements = 0;
    vector<vector<double>> matrix;

    if (rank==0) {
        int temp;
        cout << "Введите количество строк: ";
        cin >> temp;
        n = temp;

        cout << "Введите количество столбцов: ";
        cin >> temp;
        m = temp;

        cout << "Введите количество прогонов: ";
        cin >> temp;
        numMeasurements = temp;

        random_device rd;
        matrix = randomMatrix(n, m);
    }

    // Передаем значения переменных всем процессам
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numMeasurements, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // ------------------------------
    // Распределение матрицы по процессам
    // ------------------------------
    // Вычисляем распределение строк: у каждого процессa numRows, и startRow
    int base = n / size;
    int rem = n % size;
    int localRows = base + (rank < rem ? 1 : 0);
    int startRow = rank * base + min(rank, rem);

    // Подготовим локальный блок (каждый процесс хранит только его строки)
    vector<vector<double>> localBlock(localRows, vector<double>(m, 0.0));

    // Разослать блоки: root упакует всю матрицу в плоский буфер и посылает суб-блоки
    if (rank == 0) {
        // делаем плоский буфер для удобной отправки смежных участков
        vector<double> flat;
        flat.reserve((size_t)n * m);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                flat.push_back(matrix[i][j]);

        // Отправляем каждому процессу его блок (включая нулевой — просто копируем)
        for (int p = 0; p < size; ++p) {
            int pRows = base + (p < rem ? 1 : 0);
            int pStart = p * base + min(p, rem);
            int count = pRows * m;
            if (p == 0) {
                // копируем в локальный блок
                for (int i = 0; i < pRows; ++i)
                    for (int j = 0; j < m; ++j)
                        localBlock[i][j] = flat[(pStart + i) * m + j];
            } else {
                // отправляем contiguous участок из flat
                if (count > 0)
                    MPI_Send(flat.data() + (pStart * m), count, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                else {
                    // можно отправить message с count=0 (необязательно)
                    MPI_Send(nullptr, 0, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        // Рабочие получают свой плоский блок и разбирают в localBlock
        int recvCount = localRows * m;
        if (recvCount > 0) {
            vector<double> buf(recvCount);
            MPI_Recv(buf.data(), recvCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < localRows; ++i)
                for (int j = 0; j < m; ++j)
                    localBlock[i][j] = buf[i * m + j];
        } else {
            // никаких строк — localBlock пустой (size 0)
            localBlock.clear();
        }
    }

    // ------------------------------
    // Вычисления
    // ------------------------------
    
    if (rank==0) {
        cout <<endl << "Результаты " << numMeasurements << " прогонов" << " по алгоритмам:" << endl;
        measureTimeSeq(matrix, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    measureTimePar(localBlock, numMeasurements);

    MPI_Barrier(MPI_COMM_WORLD);
    measureTimeParOpt(localBlock, numMeasurements);
    
    MPI_Finalize();
    return 0;
}
