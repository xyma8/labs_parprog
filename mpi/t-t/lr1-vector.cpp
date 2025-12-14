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
    int base = n / size;
    int rem = n % size;

    int localSize = base + (rank < rem ? 1 : 0);
    int start = rank * base + min(rank, rem);

    // Локальные буферы
    vector<double> localA(localSize), localB(localSize);

    // Рассылка блоков (точка-точка)
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            int pSize = base + (p < rem ? 1 : 0);
            int pStart = p * base + min(p, rem);
            if (p == 0) {
                for (int i = 0; i < pSize; ++i) {
                    localA[i] = vecA[pStart + i];
                    localB[i] = vecB[pStart + i];
                }
            } else {
                MPI_Send(vecA.data() + pStart, pSize, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                MPI_Send(vecB.data() + pStart, pSize, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        if (localSize > 0) {
            MPI_Recv(localA.data(), localSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(localB.data(), localSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Измеряем среднее время вычисления скалярного произведения
    double totalTime = 0.0;
    double globalDot = 0.0;

    for (int r = 0; r < numMeasurements; ++r) {
        double start = MPI_Wtime();

        double localDot = 0.0;
        for (int i = 0; i < localSize; ++i)
            localDot += localA[i] * localB[i];

        // Отправляем частичные суммы
        if (rank == 0) {
            globalDot = localDot;
            double recvVal;
            for (int src = 1; src < size; ++src) {
                MPI_Recv(&recvVal, 1, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                globalDot += recvVal;
            }
        } else {
            MPI_Send(&localDot, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }

        double end = MPI_Wtime();
        totalTime += (end - start) * 1e6;
    }

    double avgTime = totalTime / numMeasurements;

    if (rank == 0)
        cout << " | Параллельный MPI: " << avgTime << " мкс; " << "dot=" << globalDot << endl;
}

// Метод подсчета времени выполнения параллельного с оптимизацией выполнения
void measureTimeParOpt(vector<double>& vecA, vector<double>& vecB, int numMeasurements) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = vecA.size();
    int base = n / size;
    int rem = n % size;

    int localSize = base + (rank < rem ? 1 : 0);
    int start = rank * base + min(rank, rem);

    // Локальные буферы
    vector<double> localA(localSize), localB(localSize);

    // Рассылка блоков (точка-точка)
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            int pSize = base + (p < rem ? 1 : 0);
            int pStart = p * base + min(p, rem);
            if (p == 0) {
                for (int i = 0; i < pSize; ++i) {
                    localA[i] = vecA[pStart + i];
                    localB[i] = vecB[pStart + i];
                }
            } else {
                MPI_Send(vecA.data() + pStart, pSize, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                MPI_Send(vecB.data() + pStart, pSize, MPI_DOUBLE, p, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        if (localSize > 0) {
            MPI_Recv(localA.data(), localSize, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(localB.data(), localSize, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Измеряем среднее время вычисления скалярного произведения
    double totalTime = 0.0;
    double globalDot = 0.0;

    for (int run = 0; run < numMeasurements; ++run) {
        double start = MPI_Wtime();

        double localDot = 0.0;
        for (int i = 0; i < localSize; ++i)
            localDot += localA[i] * localB[i];

        // pairwise reduction (логарифмическая схема)
        double recvVal;
        for (int step = 1; step < size; step <<= 1) {
            if (rank % (2 * step) == 0) {
                int partner = rank + step;
                if (partner < size) {
                    MPI_Recv(&recvVal, 1, MPI_DOUBLE, partner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    localDot += recvVal;
                }
            } else {
                int partner = rank - step;
                MPI_Send(&localDot, 1, MPI_DOUBLE, partner, 2, MPI_COMM_WORLD);
                break;
            }
        }

        if (rank == 0) globalDot = localDot;

        double end = MPI_Wtime();
        totalTime += (end - start) * 1e6; 
    }

    double avgTime = totalTime / numMeasurements;

    if (rank == 0)
        cout << " | Параллельный MPI (оптимизация): " << avgTime << " мкс; " << "dot=" << globalDot << endl;
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