#include <mpi.h>
//#include <Windows.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>  // для setw, left, right
using namespace std;
using clk = std::chrono::steady_clock; // устойчив к смене системного времени

// Функция генерации вектора n размерности
vector<double> randomVector(size_t n, unsigned int seed = 12345, double minVal = -1000.0, double maxVal = 1000.0) {
    mt19937 gen(seed); // генератор Marsenne Twister
    uniform_real_distribution<> dist(minVal, maxVal); // равномерное распределение

    vector<double> vec(n); // пустой вектор с указанием размерности
    for (size_t i = 0; i < n; ++i) {
        vec[i] = dist(gen);
    }

    return vec;
}

// Метод подсчета времени выполнения последовательного выполнения
void measureTimeSeq(vector<double>& vecA, vector<double>& vecB, int numMeasurements) {
    double seqTotalTime = 0;

    double dot = 0.0;

    for (int i = 0; i < numMeasurements; ++i) {
        dot = 0;
        auto t0 = clk::now();

        for (size_t i = 0; i < vecA.size(); ++i) {
            dot += vecA[i] * vecB[i];
        }

        auto t1 = clk::now();
        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        seqTotalTime += dt;
    }

    double seqAvgTime = seqTotalTime / numMeasurements;
    cout << left << setw(25) << " | Последовательный:"
        << left << seqAvgTime << " мкс; "
        << "dot=" << dot << endl;
}

// Метод подсчета времени выполнения параллельного выполнения
void measureTimePar(vector<double>& vecA, vector<double>& vecB, int numMeasurements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = vecA.size();

    // --- рассчитываем размер блока для каждого процесса ---
    vector<int> sendCounts(size);
    vector<int> displs(size);
    int base = n / size;
    int rem = n % size;
    for (int i = 0; i < size; ++i) {
        sendCounts[i] = base + (i < rem ? 1 : 0);
        displs[i] = i * base + min(i, rem);
    }

    // --- выделяем локальные блоки ---
    int localSize = sendCounts[rank];
    vector<double> localA(localSize), localB(localSize);

    // --- рассылка частей векторов ---
    MPI_Scatterv(vecA.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                 localA.data(), localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vecB.data(), sendCounts.data(), displs.data(), MPI_DOUBLE,
                 localB.data(), localSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- вычисление среднего времени ---
    double totalTime = 0.0;
    double dotResult = 0.0;

    for (int r = 0; r < numMeasurements; ++r) {
        double start = MPI_Wtime();

        double localDot = 0.0;
        for (int i = 0; i < localSize; ++i)
            localDot += localA[i] * localB[i];

        // --- собираем результат через редукцию ---
        MPI_Reduce(&localDot, &dotResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        double end = MPI_Wtime();
        totalTime += (end - start) * 1e6; 
    }

    double avgTime = totalTime / numMeasurements;
    if (rank == 0)
        cout << " | Параллельный MPI: " << avgTime << " мкс; " << "dot=" << dotResult << endl;
}

// Метод подсчета времени выполнения параллельного с оптимизацией выполнения
void measureTimeParOpt(vector<double>& vecA, vector<double>& vecB, int numMeasurements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n = (int)vecA.size();
    int base = n / size;
    int rem = n % size;

    int start = rank * base + min(rank, rem);
    int localSize = base + (rank < rem ? 1 : 0);

    vector<double> localA(localSize);
    vector<double> localB(localSize);

    // Рассылка блоков по точка-точка
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            int pStart = p * base + min(p, rem);
            int pCount = base + (p < rem ? 1 : 0);

            if (p == 0) {
                for (int i = 0; i < pCount; ++i) {
                    localA[i] = vecA[pStart + i];
                    localB[i] = vecB[pStart + i];
                }
            } else if (pCount > 0) {
                MPI_Send(vecA.data() + pStart, pCount, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                MPI_Send(vecB.data() + pStart, pCount, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
            }
        }
    } else if (localSize > 0) {
        MPI_Recv(localA.data(), localSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(localB.data(), localSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Измерение времени
    double totalTime = 0.0;
    double dotResult = 0.0;

    for (int r = 0; r < numMeasurements; ++r) {
        double start = MPI_Wtime();

        double localDot = 0.0;
        for (int i = 0; i < localSize; ++i)
            localDot += localA[i] * localB[i];

        // Оптимизация — бинарное дерево редукции через точка-точка
        for (int step = 1; step < size; step <<= 1) {
            if ((rank % (2 * step)) == 0) {
                int partner = rank + step;
                if (partner < size) {
                    double recvVal;
                    MPI_Recv(&recvVal, 1, MPI_DOUBLE, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    localDot += recvVal;
                }
            } else {
                int partner = rank - step;
                MPI_Send(&localDot, 1, MPI_DOUBLE, partner, 2, MPI_COMM_WORLD);
                break;
            }
        }

        if (rank == 0) dotResult = localDot;

        double end = MPI_Wtime();
        totalTime += (end - start) * 1e6; 
    }

    double avgTime = totalTime / numMeasurements;
    if (rank == 0)
        cout << " | Параллельный MPI: " << avgTime << " мкс; " << "dot=" << dotResult << endl;
}

int main(int argc, char** argv) {
    //SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    //SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = 0, numMeasurements = 0;
    vector<double> vectorA;
    vector<double> vectorB;

    if (rank==0) {
        int temp;
        cout << "Введите размерность векторов: ";
        cin >> temp;
        n = temp;

        cout << "Введите количество прогонов: ";
        cin >> temp;
        numMeasurements = temp;

        random_device rd;
        vectorA = randomVector(n, 12345);
        vectorB = randomVector(n, 54321);
    }

    // Передаем значения переменных всем процессам
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numMeasurements, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (rank != 0) {
         vectorA.resize(n); vectorB.resize(n); // чтобы передать фрагменты
    }

    // ------------------------------
    // Вычисления
    // ------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) {
        cout <<endl << "Результаты " << numMeasurements << " прогонов" << " по алгоритмам:" << endl;
        measureTimeSeq(vectorA, vectorB, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    measureTimePar(vectorA, vectorB, numMeasurements);

    MPI_Barrier(MPI_COMM_WORLD);
    measureTimeParOpt(vectorA, vectorB, numMeasurements);

    MPI_Finalize();
    return 0;
}