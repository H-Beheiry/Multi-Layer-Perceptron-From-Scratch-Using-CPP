#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <cmath>
#include <cstdlib>
using namespace std;

struct connection{
    double weight;
    double deltaWeight;
};

class Nueron;
typedef vector<Nueron> Layer;
class Nueron 
{
public:
    Nueron(unsigned numOutputs, unsigned m_index);

    void setOutputVal(double val){m_outputVal=val;}
    double getOutputVal(void) const {return m_outputVal;}

    static double activationFunc(double x);
    static double activationFuncDerivative(double x);
    void feedForward(const Layer &prevLayer);
    void calculateOutputGrads(double targetVal);
    void calculateHiddenGrad(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);


private:
    static double lr;
    static double alpha;

    double m_outputVal;
    unsigned m_index;
    double sumDOW(const Layer &nextLayer) const;
    static double randomWeight(void){return rand() / double(RAND_MAX)*2.0 -1.0;}
    vector<connection> m_outputWeights;
    double m_gradient;
};

#endif