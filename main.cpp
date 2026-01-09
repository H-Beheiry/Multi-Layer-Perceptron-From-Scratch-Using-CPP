#include <iostream>
#include <vector>
#include <ctime>
#include "Net.h"
#include "MnistCSV.h" 
#include <algorithm>
#include <chrono>
using namespace std;

int getPrediction(const vector<double>& values) {
    return distance(values.begin(), max_element(values.begin(), values.end()));
}

int main() {
    srand(time(NULL));

    try {
        MnistData data = MnistLoader::loadCSV("mnist_train.csv", 30000); 
        vector<unsigned> topology = { 784, 64, 10 };
        Net myNet(topology);

        // TRAIN
        auto startTrain = chrono::high_resolution_clock::now();

        cout << "Training on first 25000...\n";
        for (int i = 0; i < 25000; ++i) {
            myNet.feedForward(data.trainingImages[i]);
            myNet.backProp(data.trainingLabels[i]);
            
            if (i % 200 == 0) {
                 cout << "Iter: " << i << " | Error: " << myNet.getRecentAverageError() << endl;
            }
        }

        auto endTrain = chrono::high_resolution_clock::now();
        chrono::duration<double> trainTime = endTrain - startTrain;

        // TEST

        auto startTest = chrono::high_resolution_clock::now();

        cout << "\nTesting on last 5000\n";
        int correct = 0;
        for (int i = 25000; i < 30000; ++i) {
            myNet.feedForward(data.trainingImages[i]);
            
            vector<double> results;
            myNet.getResults(results);

            if (getPrediction(results) == getPrediction(data.trainingLabels[i])) {
                correct++;
            }
        }
        auto endTest = chrono::high_resolution_clock::now();
        chrono::duration<double> testTest = endTest - startTest;
        cout << "Training Time: " << trainTime.count() << " seconds\n";
        cout << "Testing Time:  " << testTest.count() << " seconds\n";
        cout << "Accuracy: " << (double)correct / 5000.0 * 100.0 << "%\n";





    } catch (const exception &e) {
        cerr << "Error: " << e.what();
    }

    return 0;

}