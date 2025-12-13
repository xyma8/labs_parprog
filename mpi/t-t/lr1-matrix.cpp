#include <iostream>
//#include <Windows.h>
#include <random>
#include <chrono>
#include <vector>
#include <mpi.h>
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

        static volatile double sink; // защищаем от выкидывания
        sink = maxv;

        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        seqTotalTime += dt;
    }

    double seqAvgTime = seqTotalTime / numMeasurements;
    cout << " | Последовательный: " << seqAvgTime << " мкс; " << "max=" << maxv << endl;
}

// Метод подсчета времени выполнения параллельного выполнения
void measureTimePar(vector<vector<double>>& matrix, int numMeasurements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = matrix.size();
    int m = matrix[0].size();

    double totalTime = 0.0;
    double globalMax = -numeric_limits<double>::infinity();

    for (int r = 0; r < numMeasurements; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        int rowsPerProc = n / size;
        int remainder = n % size;

        // Размер куска для каждого процесса
        int startRow = rank * rowsPerProc + min(rank, remainder);
        int endRow = startRow + rowsPerProc + (rank < remainder ? 1 : 0);

        // Вычисляем локальный максимум
        double localMax = -numeric_limits<double>::infinity();
        for (int i = startRow; i < endRow; ++i)
            for (double val : matrix[i])
                if (val > localMax)
                    localMax = val;

        if (rank != 0) {
            MPI_Send(&localMax, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        } else {
            globalMax = localMax;
            for (int src = 1; src < size; ++src) {
                double recvVal;
                MPI_Recv(&recvVal, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (recvVal > globalMax)
                    globalMax = recvVal;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();
        totalTime += (end - start) * 1e6; // в микросекундах
    }

    double avgTime = totalTime / numMeasurements;

    if (rank == 0)
        cout << " | Параллельный MPI: " << avgTime << " мкс; " << "max=" << globalMax << endl;
}

void measureTimeParOptimized(vector<vector<double>>& matrix, int numMeasurements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = matrix.size();
    int m = matrix[0].size();

    // Распределяем строки
    int rowsPerProc = n / size;
    int remainder = n % size;
    int startRow = rank * rowsPerProc + min(rank, remainder);
    int endRow = startRow + rowsPerProc + (rank < remainder ? 1 : 0);
    int localRows = endRow - startRow;

    vector<double> localData(localRows * m);
    if (rank == 0) {
        // Рассылка кусков вручную (точка-точка)
        for (int dest = 1; dest < size; ++dest) {
            int s = dest * rowsPerProc + min(dest, remainder);
            int e = s + rowsPerProc + (dest < remainder ? 1 : 0);
            int count = (e - s) * m;
            MPI_Send(matrix[s].data(), count, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        }
        // Копируем свой кусок
        for (int i = 0; i < localRows; ++i)
            copy(matrix[startRow + i].begin(), matrix[startRow + i].end(),
                 localData.begin() + i * m);
    } else {
        MPI_Recv(localData.data(), localRows * m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double totalTime = 0.0;
    double localMax = -numeric_limits<double>::infinity();
    for (int r = 0; r < numMeasurements; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        // Локальный максимум
        localMax = -numeric_limits<double>::infinity();
        for (double val : localData)
            if (val > localMax) localMax = val;

        // Двоичное дерево обменов (точка-точка)
        double recvVal;
        for (int step = 1; step < size; step *= 2) {
            if (rank % (2 * step) == 0) {
                int src = rank + step;
                if (src < size) {
                    MPI_Recv(&recvVal, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (recvVal > localMax) localMax = recvVal;
                }
            } else {
                int dest = rank - step;
                MPI_Send(&localMax, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
                break;
            }
        }

        double end = MPI_Wtime();
        totalTime += (end - start) * 1e6;
    }

    if (rank == 0) {
        double avgTime = totalTime / numMeasurements;
        cout << " | Оптимизированный MPI (точка-точка): " << avgTime << " мкс; max=" << fixed << localMax << endl;
    }
}


// Метод подсчета времени выполнения параллельного с оптимизацией выполнения
void measureTimeParOpt(vector<vector<double>>& matrix, int numMeasurements) {
    double parTotalTimeOpt = 0;
    double maxv = -numeric_limits<double>::infinity();
    pair<size_t, size_t> size = getMatrixSize(matrix);

    // Многократные замеры для параллельного алгоритма
    for (int i = 0; i < numMeasurements; ++i) {
        maxv = -numeric_limits<double>::infinity();
        //double t0 = omp_get_wtime();

        for (int i = 0; i < size.first; i++) {
            for (int j = 0; j < size.second; j++) {
                if (matrix[i][j] > maxv) {
                    maxv = matrix[i][j];
                }
            }
        }

        //double dt = (omp_get_wtime() - t0) * 1e6; // умножаем для микросекунд
        //parTotalTimeOpt += dt;
        //static volatile double sink; sink = maxv; // не даём выкинуть
    }
    
    double parAvgTimeOpt = parTotalTimeOpt / numMeasurements;
    cout << " | Параллельный OpenMP оптимизированный: " << parAvgTimeOpt << " мкс; " << "max=" << maxv << endl;
}

int main(int argc, char** argv)
{
    //SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    //SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
        matrix = randomMatrix(n, m, rd());
    }

    // Передаем значения переменных всем процессам
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numMeasurements, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Рассылаем матрицу всем (каждый процессу нужна копия, тк небольшие накладные)
    if (rank != 0)
        matrix.assign(n, vector<double>(m, 0.0));

    for (size_t i = 0; i < n; ++i)
        MPI_Bcast(matrix[i].data(), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    if (rank==0) {
        cout <<endl << "Результаты " << numMeasurements << " прогонов" << " по алгоритмам:" << endl;
        measureTimeSeq(matrix, numMeasurements);
    }

    measureTimePar(matrix, numMeasurements);

    measureTimeParOptimized(matrix, numMeasurements);
    MPI_Finalize();
    return 0;
}
