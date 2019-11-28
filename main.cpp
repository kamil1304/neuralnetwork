/*
I wrote fully connected Neural Netowrk for my own deeper understanfing of the forward and backward propagation.
It solves multiclassification problem of  295 long row vector of data which depicts at most 6 different classes - it is abstarct problem.
The code was written for testing my approach and for learning hence I use few const variables in the code.
The code is an example of my fast prototyping rather than production code.
*/

#include <map>
#include <set>
#include <list>
#include <cmath>
#include <ctime>
#include <deque>
#include <queue>
#include <stack>
#include <string>
#include <bitset>
#include <cstdio>
#include <limits>
#include <vector>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include <iterator>

using namespace std;
int epochs =35;
extern int num_of_data = 294;
extern string end_CVS =  ".csv";

template<class T=double>
void readCSV(const string name, vector<vector<T>> &data, vector<vector<T>> &classes) {
    fstream fin;
    fin.open(name, ios::in);

    vector<T> row_data;
    vector<T> row_class;

    string line, word, temp;
    while (getline(fin, line)) {
        stringstream s(line);

        int t = 0;
        while (t < num_of_data) {
            t++;
            getline(s, word, ',');
            row_data.push_back(atof(word.c_str()));
        }
        data.push_back(row_data);
        row_data.clear();

        while (getline(s, word, ',')) {
            row_class.push_back(atoi(word.c_str()));
        }
        classes.push_back(row_class);
        row_class.clear();

    }
}

template<class T=double>
void disp(const vector<vector<T>> output) {
    int num = 0;
    for (vector<T> a:output) {
        //  cout<<num<<" ";
        num++;
        for (T b:a) {
            cout << b << " ";
        }
        cout << endl;
    }
    cout << endl;

}

template<class T=double>
void saveData(const vector<vector<T>> classes, const string path) {

    fstream fout;
    fout.open(path, ios::out | ios::app);
    string row_string;

    for (int i=0; i<classes.size(); i++) {
        for (int j=0; j<classes[i].size(); j++) {
            row_string.append(to_string(classes[i][j]));
            row_string.append(",");
        }
        row_string.pop_back();
        row_string.append("\n");

        fout << row_string;
        row_string.clear();
    }
}

template<class T=double>
class Layer {
private:
    int x;
    int y;
    vector<vector<T>> l;
public:
    Layer<T>() {
        x = 0;
        y = 0;
        vector<T> a;
        l.push_back(a);
    }

    Layer(vector<vector<T>> in) {
        l = in;
        x = in.size();
        y = in[0].size();
    }

    Layer<T>(int x, int y) {
        this->x = x;
        this->y = y;
        this->l = vector<vector<T>>(x, vector<T>(y, 0.0));
    }
    inline int getX() const {return x;}
    inline int getY() const {return y;}
    inline vector<vector<T>> getL() const {return l;}

    void reset(const int x,const int y);
    void fillWithRand();
    void addOne();

    void sigmoid(const int h=0);
    void dis() { disp(this->l); }
    void saveLayerToFile(const string name);

    Layer<T> rone();
    Layer<T> transpose();
    Layer<T> elementwise(Layer<T> const &L);
    Layer<T> sigdiv();
    template<class U>
    Layer<T> operator+(const U &i);
    Layer<T> operator+(const Layer<T> &i);
    Layer<T> operator+(const vector<T> &d);
    Layer<T> operator-(const Layer<T> &t);
    Layer<T> operator-(const vector<T> &d);
    Layer<T> operator*(Layer<T> const &L);

};
template<class T>
void Layer<T>:: sigmoid(const int h) {

    for (int i=h;i<x; i++)
        for (int j=0; j<y; j++)
            l[i][j] = 1.0 / (1.0 + exp(-l[i][j]));
}

template<class T>
void Layer<T>:: addOne() {

    if (x == 1) {
        l[0].insert(l[0].begin(), 1.0);
        y++;
    } else if (y == 1) {
        l.insert(l.begin(), vector<T>(1.0));
        x++;
    } else {
        cout << "It is not row or column vector" << endl;
    }
}

template<class T>
void Layer<T>::fillWithRand() {
    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
            l[i][j] = (((double) rand() / (RAND_MAX / 2)) - 1.0);
}

template<class T>
void Layer<T>:: reset(const int x, const int y) {
    this->x = x;
    this->y = y;
    l.clear();
    for (int i = 0; i < x; i++) {
        vector<T> row(y);
        std::fill(row.begin(), row.end(), 0.0);
        l.push_back(row);
    }
}

template<class T>
void Layer<T>::saveLayerToFile(const string file_name){

    ofstream output_file(file_name);
    ostream_iterator<T> output_iterator(output_file, ",");
    for ( int i = 0 ; i < l.size() ; i++ ){
        copy(l.at(i).begin(), l.at(i).end(), output_iterator);
        output_file<<endl;
    }
}

template<class T=double>
Layer<T> readLayer(const string path){

    fstream fin;
    fin.open(path, ios::in);

    vector<T> row_data;
    vector<vector<T>> layer;
    string line, word;
    while (getline(fin, line)) {
        stringstream s(line);

        while (getline(s, word, ',')) {
            row_data.push_back(atof(word.c_str()));
        }
        layer.push_back(row_data);
        row_data.clear();
    }
    Layer<T> out(layer);
    return out;
}

template<class T>
Layer<T> Layer<T>::operator+(const Layer<T> &lay){

    Layer<T> out= *this;

    for(int i=0; i<out.getX(); i++){
        for(int j=0; j<out.getY(); j++){
            out.getL()[i][j]=out.getL()[i][j]+lay.getL()[i][j];
        }
    }
    return out;
}


template<class T>
Layer<T> Layer<T>::sigdiv(){

    Layer<T> out =*this;
    for(vector<T>& h:this->l)
        for(T &g:h)
            g=g*(1.0-g);

    return out;
}

template<class T>
Layer<T> Layer<T>::rone() {

    vector<vector<T>> temp=this->getL();
    int x=this->getX();
    for (int i = 0; i < x; i++)
        temp[i].erase(this->getL()[i].begin());

    return Layer(temp);
}

template<class T>
Layer<T> Layer<T>::elementwise(const Layer<T> &l2) {

    try {
        if (this->x != l2.getX() || this->y != l2.getY())
            throw 10;
    }
    catch (int) {
        cout << "wrong dimensions in elementwise multiplication, first: "<<this->x<<" "<<this->y;
        cout<<" second: "<<l2.getX()<<" "<<l2.getY()<<endl;
    }

    Layer<T> out = l2;
    for (int i = 0; i < l2.getX(); i++) {
        for (int j = 0; j < l2.getY(); j++) {
            out.getL()[i][j] = l2.getL()[i][j] * this->l[i][j];
        }
    }
    return out;
}

template<class T>
Layer<T> Layer<T>::operator*(const Layer<T> &layer2) {
    int m1 = this->x;
    int m2 = this->y;
    int n1 = layer2.getX();
    int n2 = layer2.getY();

    try {
        if (m2 != n1)
            throw 10;
    } catch (int ) {
        cout << "Wrong dimensions; " << endl;
        cout << "layer1: " << m1 << " " << m2 << endl;
        cout << "layer2: " << n1 << " " << n2 << endl;

    }
    Layer res = layer2;
    res.reset(m1, n2);
    for (int i = 0; i < m1; i++) {

        vector<T> column;
        for (int j = 0; j < n2; j++) {
            vector<T> column;
            for (int k = 0; k < n1; k++) {
                column.push_back(layer2.getL()[k][j]);
            }
            res.getL()[i][j] = std::inner_product(this->l[i].begin(), this->l[i].end(), column.begin(), 0.0);
        }
    }
    return res;
}

template<class T>
template<class U>
Layer<T> Layer<T>::operator+(const U &i) {
    Layer temp = *this;
    for (auto &a:temp.getL()) {
        for (auto &v:a) {
            v += T(i);
        }
    }
    return temp;
}

template<class T>
Layer<T> Layer<T>::operator-(const Layer<T> &t) {

    Layer res = *this;
    try {
        if (this->x == t.getX() && this->y == t.getY()) {
            for (int i = 0; i < x; i++) {
                for (int j = 0; j < y; j++) {
                    res.getL()[i][j] -= t.getL()[i][j];
                }
            }
        } else
            throw 10;
    } catch (int) {
        cout << "Vectors have different dimesnions \n";
    }
    return res;
}

template<class T>
Layer<T> Layer<T>::operator-(const vector<T> &d) {

    Layer res = *this;
    if (d.size() == 3) {
        cout << "Vector wrong size" << endl;
        return res;
    }

    res.getL()[d[1]][d[2]] -= d[0];
    return res;
}

template<class T>
Layer<T> Layer<T>::operator+(const vector<T> &d) {

    Layer res = *this;
    if (d.size() == 3) {
        cout << "Vector wrong size" << endl;
        return res;
    }

    res.getL()[d[1]][d[2]] += d[0];
    return res;
}

template<class T>
Layer<T> Layer<T>::transpose() {
    vector<vector<T>> outtrans(y, vector<T>(x));
    for (size_t i = 0; i < x; ++i)
        for (size_t j = 0; j < y; ++j)
            outtrans[j][i] = this->l[i][j];
    return Layer<T>(outtrans);
}

template<class T=double>
Layer<T> operator*( Layer<T> layer,double rate){

    return rate*layer;
}

template<class T=double>
Layer<T> operator*(double rate, Layer<T> layer){
    Layer<T> out =layer;

    for(vector<T> &t:out.getL())
        for(T& a:t)
            a= rate*a;

    return out;
}
template<class T=double>
class NN {
private:
    vector<Layer<T>> weights;
    vector<Layer<T>> layers;
    vector<Layer<T>> deltas;
public:
    Layer<T> inputLayer;
    inline vector<Layer<T>> getLayers() const{return layers;}
    inline vector<Layer<T>> getWeights() const{return weights;}
    inline vector<Layer<T>> getDeltas() const{return deltas;}

    NN<T>();
    NN<T>(const string name);

    void forwardProp(Layer<T> &input);
    void backwardProp(Layer<T> &expected);
    void weightUpdate(double);
    void save(const string path);
    Layer<T> predict(Layer<T> input);
};
template<class T>
void NN<T>::save(const string path ){

  for(int i =0; i<3; i++)
      this->weights.at(i).saveLayerToFile(path+to_string(i)+end_CVS);
}

template<class T>
Layer<T> NN<T>::predict(Layer<T> input) {

    this->layers[0] = input * this->weights[0];
    this->layers[0].sigmoid();
    this->layers[0].addOne();

    this->layers[1]=this->layers[0]*this->weights[1];
    this->layers[1].sigmoid();
    this->layers[1].addOne();

    this->layers[2] = this->layers[1] * this->weights[2];
    this->layers[2].sigmoid();

    return this->layers[2];
}
template<class T>
NN<T>::NN() {

    Layer<T> weights1(294 + 1, 100);
    Layer<T> layer2(1, 100);

    Layer<T> weights21(100+1, 100);
    Layer<T> layer21(1, 100);

    Layer<T> weights2(100 + 1, 6);
    Layer<T> layer3(1, 6);

    weights1.fillWithRand();
    weights21.fillWithRand();
    weights2.fillWithRand();

    weights.push_back(weights1);
    weights.push_back(weights21);
    weights.push_back(weights2);

    layers.push_back(layer2);
    layers.push_back(layer21);
    layers.push_back(layer3);
}
template<class T>
NN<T>::NN(const string name) {

    Layer<T> layer2(1, 100);
    Layer<T> layer21(1, 100);
    Layer<T> layer3(1, 6);

    for(int i=0; i<3; i++)
        weights.push_back(readLayer(name+to_string(i)+end_CVS));

    layers.push_back(layer2);
    layers.push_back(layer21);
    layers.push_back(layer3);
}

template<class T>
void NN<T>::forwardProp(Layer<T> &input) {
    inputLayer=input;
    this->layers[0] = input * this->weights[0];
    this->layers[0].sigmoid();
    this->layers[0].addOne();

    this->layers[1]=this->layers[0]*this->weights[1];
    this->layers[1].sigmoid();
    this->layers[1].addOne();

    this->layers[2] = this->layers[1] * this->weights[2];
    this->layers[2].sigmoid();
}

template<class T>
void NN<T>::backwardProp(Layer<T> &expected) {

    deltas=    vector<Layer<T>> (layers.size());

    for (int j = layers.size() - 1; j > -1; j--) {

        if (j == layers.size() - 1) {

            try {
                Layer<T> results = expected;
                results = expected - layers.back();
                deltas[j] = results;
            }
            catch (int) {
                cout << "Not all good _ first back" << endl;
            }

        } else if (j == layers.size() - 2) {
            try {
           //     cout<<j<<" "<<weights[j+1].getX()<<" "<<weights[j+1].getY()<<"   D:"<<deltas[j+1].transpose().getX()<<" "<<deltas[j+1].transpose().getY()<<endl;
                deltas[j] = (weights[j+1] * deltas[j + 1].transpose()).transpose();
                deltas[j] = deltas[j].elementwise(layers[j].sigdiv());
            }
            catch (int) {
                cout << "Not all good _ second back" << endl;
            }

        } else {
            try {
                //     Layer<T> a =(deltas[j + 1].rone());
                //    cout << j << " " << weights[j+1].getX() << " " << weights[j+1].getY() << "   D:"
                //         << a.getX() << " " << a.getY() << endl;
                deltas[j] = (weights[j+1] * (deltas[j + 1].rone()).transpose()).transpose();
                deltas[j]=deltas[j].elementwise(layers[j].sigdiv());
            }
            catch (int) {
                cout << "Not all good _ second back" << endl;

            }

        }
    }
}

template<class T=double>
double cost(const Layer<T> last, const Layer<T> o) {

    double cost = 0;

    for (int i = 0; i < last.getX(); i++) {
        for (int j = 0; j < last.getY(); j++) {
            cost += (o.getL()[i][j] * log(last.getL()[i][j]) + (1.0 - o.getL()[i][j]) * (1.0 - log(last.getL()[i][j])));

        }
    }
    return cost;
}

template<class T>
void NN<T>::weightUpdate(double rate){


    weights[0] = weights[0]+rate* ((inputLayer.transpose())*(deltas[0].rone()));

    for(int i=0; i<weights.size()-1; i++){
        auto c =  ((layers[i].transpose())*(deltas[i+1].rone()));
        weights[i+1] = weights[i+1]+rate* ((layers[i].transpose())*(deltas[i+1].rone()));
    }

}
template<class T=double>
void readLayers(const string path, vector<Layer<T>> &inputs, vector<Layer<T>> &expected){

    vector<vector<T>> data_train;
    vector<vector<T>> classes_train;

    readCSV(path, data_train, classes_train);

    for (int i = 0; i < data_train.size(); i++) {

        Layer<T> temp1(vector<vector<T>>(1, data_train[i]));
        temp1.addOne();

        Layer<T> temp2(vector<vector<T>>(1, classes_train[i]));

        inputs.push_back(temp1);
        expected.push_back(temp2);
    }
}

void testNN(const string read_weights, const string path_training){

  NN nn2(read_weights);

  vector<Layer<double>> inputs_test;
  vector<Layer<double>> expected_test;
  readLayers( path_training, inputs_test, expected_test);
  vector<vector<double>> out;
  for(int i =0; i<inputs_test.size();i++) {
      Layer<double> t = nn2.predict(inputs_test[i]);
      out.push_back(t.getL()[0]);
  }
  string path_out = "/home/kamil/Desktop/ad_task/test_out_answers.csv";
  saveData(out, path_out);

  vector<vector<double>> expected_answers;
  for(int i=0; i<expected_test.size(); i++)
      expected_answers.push_back(expected_test[i].getL()[0]);

  string path_out_expected = "/home/kamil/Desktop/ad_task/test_out_answers_expected.csv";
  saveData(expected_answers, path_out_expected);

  string path_out_costs = "/home/kamil/Desktop/ad_task/test_out.csv";
  //vector<vector<double>> out_costs;
//  out_costs.push_back(costs);
//  saveData(out_costs, path_out_costs);
}

int main() {
    srand(time(NULL));
    string path_training= "/home/kamil/Desktop/ad_task/train.csv";
    string path_testing= "/home/kamngil/Desktop/ad_task/test.csv";
    string path_save_model = "/home/kamil/Desktop/ad_task/test_save";
    string read_weights = "/home/kamil/Desktop/ad_task/test_save";

    vector<Layer<double>> inputs;
    vector<Layer<double>> expected;
    readLayers( path_training, inputs, expected);

    NN nn;
    double costSum =0;
    vector<double> costs;

    for (int e=0; e<epochs; e++) {
        costSum=0;
        cout<<"Epoch number: "<<e<<endl;
        double l=inputs.size();
        for (int i = 0; i < int(l); i++) {

            nn.forwardProp(inputs[i]);
            double temp = cost(nn.getLayers().back(), expected[i]);
            nn.backwardProp(expected[i]);
            nn.weightUpdate(0.001);
            costSum-=temp;
        }
        cout << "Cost: " << costSum/l << endl;
        costs.push_back(costSum/l);
    }

    nn.save(path_save_model);
    testNN(read_weights,path_training);

  return 0;
}
