#include "Neuron.h"

double Nueron::lr=0.15;
double Nueron::alpha=0.5;

Nueron::Nueron(unsigned numOutput, unsigned index){
    for (unsigned c=0; c<numOutput; c++){
        m_outputWeights.push_back(connection());
        m_outputWeights.back().weight= randomWeight();
    }
    m_index= index;
}

double Nueron::activationFunc(double x){
    // return tanh(x);
    return 1.0 / (1.0 + exp(-x));
}
double Nueron::activationFuncDerivative(double x){
    // return 1.0 - x * x;
    return x * (1.0 - x);
}

void Nueron::updateInputWeights(Layer &prevLayer){
    for (unsigned n=0; n<prevLayer.size();++n){
        Nueron &nueron=prevLayer[n];
        double oldDeltaW= nueron.m_outputWeights[m_index].deltaWeight;
        double newDeltaW= 
            lr*
            nueron.getOutputVal()*
            m_gradient+
            alpha*oldDeltaW;
        nueron.m_outputWeights[m_index].deltaWeight= newDeltaW;
        nueron.m_outputWeights[m_index].weight+= newDeltaW;
    }
}

double Nueron::sumDOW(const Layer &nextLayer) const{
    double sum= 0.0;
    for (unsigned n=0;n<nextLayer.size()-1;++n){
        sum+=m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Nueron::calculateHiddenGrad(const Layer &nextLayer){
    double dow= sumDOW(nextLayer);
    m_gradient= dow*Nueron::activationFuncDerivative(m_outputVal);
}


void Nueron::calculateOutputGrads(double targetVal){
    double delta= targetVal-m_outputVal;
    m_gradient= delta*Nueron::activationFuncDerivative(m_outputVal);
}

void Nueron :: feedForward(const Layer &prevLayer){
    double sum= 0;
    for (unsigned n=0; n<prevLayer.size(); ++n){
        sum+= prevLayer[n].getOutputVal()* prevLayer[n].m_outputWeights[m_index].weight;
    }
    m_outputVal= Nueron::activationFunc(sum);
}

