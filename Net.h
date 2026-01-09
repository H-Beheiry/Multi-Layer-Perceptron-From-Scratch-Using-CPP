#ifndef NET_H
#define NET_H
#include <vector>
#include "Neuron.h"
using namespace std;

class Net 
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError();
    
private:
    vector <Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
    
};

#endif