#include "learned_index.h"
#include <iostream>
using namespace std;

int main(int argc,char *argv[]){

  RMIConfig rmi_config;
  RMIConfig::StageConfig first, second;

  first.model_type = RMIConfig::StageConfig::LinearRegression;
  first.model_n = 1;

  
  second.model_n = atoi(argv[1]);
  second.model_type = RMIConfig::StageConfig::LinearRegression;
  rmi_config.stage_configs.push_back(first);
  rmi_config.stage_configs.push_back(second);

  LearnedRangeIndexSingleKey<uint64_t,float> table(rmi_config);

  srand((unsigned)time(NULL)); 
  vector<double> x;
  vector<double> y;
  double key = 0;
  double value = 0;

  std::ifstream input_file("result.txt");

	while (true) {
    if (!(input_file >> value)) break;
    key += rand() % 100 + 3;
    // cout << "train:" << key << ": " << value << endl;
    table.insert(key,value);
    x.push_back(key);
    y.push_back(value);
  }

  table.finish_insert();
  table.finish_train();

  vector<int> result(15000);
  for (int i = 0; i < x.size(); i++){
    key = x[i];
    value = y[i];
    auto value_get = table.get(key);
    double bit = 1.0 * (value-value_get) / value;
    int block = value_get / 4096;
    // cout << i << " result: " << value_get << " : " << value << "; block:" << block <<
    //       "; error:" << (value-value_get)  << ";error bit: "<< bit << endl;
    result[block]++;
  }
  // for (int i = 0; i < result.size(); i++){
  //   if (result[i] != 0)
  //     cout << i << " block_num: " << result[i] << endl;
  // }

  table.printR();

  // serialize && deserialize
  string param;
  table.serialize(param);
  // cout << "serialize: " << param << " ;lenth: " << param.length() << endl;
  cout << "serialize lenth: " << param.length() << endl;

  LearnedRangeIndexSingleKey<uint64_t,float> Rtable(param, rmi_config);
  int find=0, no_find=0;
  for (int i = 0; i < x.size(); i++){
    auto value_get = table.get(x[i]);
    auto Rvalue_get = Rtable.get(x[i]);
    
    if (value_get != Rvalue_get){
      cout << i << ": value_get( " << value_get <<  " )!= Rvalue_get( " << Rvalue_get << " )" << endl;
      no_find ++;
    }
    else{
      cout << i << ": value_get( " << value_get <<  " )== Rvalue_get( " << Rvalue_get << " )" << endl;
      find ++;
    }
  }
  cout << "total: " << x.size() << " ;right: " << find << " ;wrong: " << no_find << endl;
} // end namespace test
