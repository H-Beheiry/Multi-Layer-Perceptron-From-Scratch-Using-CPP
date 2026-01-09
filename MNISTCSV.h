#ifndef MNISTCSV_H
#define MNISTCSV_H
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
using namespace std;

struct MnistData {
    vector<vector<double>> trainingImages;
    vector<vector<double>> trainingLabels;
};

class MnistLoader {
public:
    static MnistData loadCSV(string filename, int limit = 0) {
        MnistData data;
        ifstream file(filename);
        
        if (!file.is_open()) {
            throw runtime_error("Could not open file: " + filename);
        }

        string line;
        // Skip the header line (label, pixel0, pixel1...)
        getline(file, line); 

        int count = 0;
        cout << "Loading CSV data..." << endl;

        while (getline(file, line)) {
            if (limit > 0 && count >= limit) break;

            stringstream ss(line);
            string val;
            
            // 1. Get the Label (first number in the row)
            getline(ss, val, ',');
            int label = stoi(val);

            // Create One-Hot encoded target (e.g., 3 -> [0,0,0,1,0...])
            vector<double> target(10, 0.0);
            target[label] = 1.0;
            data.trainingLabels.push_back(target);

            // 2. Get the Pixels (remaining 784 numbers)
            vector<double> image;
            while (getline(ss, val, ',')) {
                // Normalize 0-255 -> 0.0-1.0
                image.push_back(stod(val) / 255.0);
            }
            data.trainingImages.push_back(image);
            count++;
        }
        
        cout << "Loaded " << count << " samples." << endl;
        return data;
    }
};

#endif