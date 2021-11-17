#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "mkl_lapacke.h"
#include "model.h"

#define LRfirst
#define AUG

#if !defined(RMI_H)
#define RMI_H

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

class LRStage {
 public:
  LRStage(unsigned model_n) {
    for (unsigned model_i = 0; model_i < model_n; ++model_i) {
      models.emplace_back();
    }
  }

  LRStage(const std::vector<std::string>& stages) {
    for (const auto& s : stages) {
      models.push_back(LinearRegression::deserialize_hardcore(s));
    }
  }

  inline void prepare(const std::vector<double>& keys,
                      const std::vector<learned_addr_t>& indexes,
                      unsigned model_i, double& index_pred_max,
                      double& index_pred_min) {
    models[model_i].prepare(keys, indexes, index_pred_max, index_pred_min);
  }

  inline void prepare_last(const std::vector<double>& keys,
                           const std::vector<learned_addr_t>& indexes,
                           unsigned model_i) {
    if (!models[model_i].prepare_last(keys, indexes)) {
      // printf("[!!!!] model %u has 0 key\n",model_i);
    }
  }

  inline double predict(const double key, unsigned model_i) {
    auto res = models[model_i].predict(key);
    // printf("model_i: %u, predict: %f to %f\n",model_i,key,res);
    return res;
  }

  inline void predict_last(const double key, learned_addr_t& pos, unsigned model_i) {
    models[model_i].predict_last(key, pos);
  }

  inline unsigned get_model_n() const { return models.size(); }

  void reset_data() {
    data_in = std::vector<
        std::pair<std::vector<double>, std::vector<learned_addr_t>>>(
        get_model_n());
  }

  void assign_data(const double key, const unsigned index,
                   const unsigned model_i) {
    data_in[model_i].first.push_back(key);
    data_in[model_i].second.push_back(index);
  }

  std::vector<std::pair<std::vector<double>, std::vector<learned_addr_t>>>
      data_in;  // valid during preparing stages
  // private:
  std::vector<LinearRegression> models;
};

struct RMIConfig {
  struct StageConfig {
    unsigned model_n;
    enum model_t {
      LinearRegression,
      NeuralNetwork,
      BestMapModel,
      MixTopModel,
      Unknown
    } model_type;
    struct {
      int depth, width;
      std::string weight_dir;
    } nn_config;
    struct {
      int depth, width;
      std::string weight_dir;
      std::vector<std::pair<double, double>> nn_ranges;
    } mix_top_config;
#ifdef SCALE_HOT_PART
    std::vector<std::pair<double, double>> hot_parts;
#endif
  };

  std::vector<StageConfig> stage_configs;

  friend std::ostream& operator<<(std::ostream& output, const RMIConfig& p) {
    output << "RMI uses " << p.stage_configs.size() << " stages." << std::endl;
    for (uint i = 0; i < p.stage_configs.size(); ++i) {
      output << "stage #" << i
             << " uses parameter: " << p.stage_configs[i].model_n << std::endl;
    }
    return output;
  }
};

template <class Weight_T>
class RMINew {
 public:
  RMINew(const RMIConfig& config_) : config(config_) {
    assert(config.stage_configs.size() == 2);
    assert(config.stage_configs.front().model_n == 1);
    assert(config.stage_configs[1].model_type ==
           RMIConfig::StageConfig::LinearRegression);
    // printf("rmi init with model num: %u\n", config.stage_configs[1].model_n);

    // init models stage by stage
    first_stage = new LRStage(config.stage_configs[0].model_n);
    second_stage = new LRStage(config.stage_configs[1].model_n);
  }

  RMINew(const std::vector<std::string>& first,
         const std::vector<std::string>& second) {
    /**
     * XD: current RMINew only supports LR serialization.
     */
    assert(first.size() == 1);
    first_stage = new LRStage(first);
    second_stage = new LRStage(second);
  }

  RMINew(const std::vector<std::string>& first, const RMIConfig& config_) {
    first_stage = new LRStage(first);
    // printf("second stage num %d\n", config.stage_configs[1].model_n);
    second_stage = new LRStage(config_.stage_configs[1].model_n);
  }

  RMINew(const std::string& stages, const RMIConfig& config_) {

    // std::cout << __func__ << " stages size:" << stages.length() << std::endl;
    int len = config_.stage_configs[1].model_n;
    
    int size = 2 * sizeof(double);
    // int size = (stages.length()-sizeof(key_n)+1) / len;
    // std::cout << "size: " << size  << std::endl;
    // std::cout << "size: " << size << " ; total: " << stages.length()-sizeof(key_n) << std::endl;
    int pos = 0;
    
    std::vector<std::string> first;
    std::vector<std::string> second;
    
    first.push_back(stages.substr(pos, size));
    pos += size;
    for (int i = 0; i < len; i++){
      second.push_back(stages.substr(pos, size));
      pos += size;
    }
    // std::cout << "pos: " << pos << " ;sizeof(key_n): " << sizeof(key_n) << std::endl;
    // int key_len = stages.length()-pos;
    // int key_len = sizeof(key_n);
    std::string key_num = stages.substr(pos, sizeof(key_n));

    memcpy(&key_n, key_num.data() ,sizeof(key_n));
    // std::cout << "after key_n: " << key_n << std::endl;
    first_stage = new LRStage(first);
    second_stage = new LRStage(second);
  }

  RMINew(const std::string& stages, const RMIConfig& config_, unsigned num) {
    key_n = num;
    int len = config_.stage_configs[1].model_n;
    int size = stages.length() / len;
    int pos = 0;
    
    std::vector<std::string> first;
    std::vector<std::string> second;

    first.push_back(stages.substr(pos, size));
    pos += size;
    for (int i = 0; i < len; i++){
      second.push_back(stages.substr(pos, size));
      pos += size;
    }
    first_stage = new LRStage(first);
    second_stage = new LRStage(second);
  }

  ~RMINew() {
    delete first_stage;
    delete second_stage;
  }

  RMINew(const RMINew&) = delete;
  RMINew(RMINew&) = delete;

  void insert(const double key) { all_keys.push_back(key); }

  void insert(const double key, const double value) {
    all_values.push_back({key, value});
  }

  void insert_w_idx(const double key, learned_addr_t addr) {
    all_keys.push_back(key);
    all_addrs.push_back(addr);
  }

  inline double get_key(uint64_t i) {
    return first_stage->data_in.front().first[i];
  }

  inline uint32_t get_index(uint64_t i) {
    return first_stage->data_in.front().second[i];
  }

  // bool cmp1(std::pair<int, int> a, std::pair<int, int> b) { return a.first < b.first; }

  void finish_insert(bool train_first_layer = true) {
    if (all_values.empty()) return;

    key_n = all_values.size();
    // std::cout << "finish_insert key_n:" << key_n << std::endl;
    struct myclass {
      bool operator()(std::pair<double, double> i,
                      std::pair<double, double> j) {
        return i.first < j.first;
      }
    } my_comparitor;
    sort(all_values.begin(), all_values.end(), my_comparitor);
    // printf("finish insert with: %u keys\n", key_n);

    // feed all data to the only model in the 1st stage
    first_stage->reset_data();
    second_stage->reset_data();
    for (int i = 0; i < key_n; ++i) {
      first_stage->data_in.front().first.push_back(all_values[i].first);
      first_stage->data_in.front().second.push_back(all_values[i].second);
    }

    assert(first_stage->get_model_n() == 1);
    // prepare 1st stage model with fed in data
    for (int model_i = 0; model_i < first_stage->get_model_n(); ++model_i) {
      std::vector<double>& uni_keys = first_stage->data_in[model_i].first;
      std::vector<learned_addr_t>& uni_indexes =
          first_stage->data_in[model_i].second;

      // else, dispatch data to the next stage after preparing
      // COUT_THIS("normal-first_stage");
      if (train_first_layer) {
        first_stage->prepare(uni_keys, uni_indexes, model_i,
                             first_stage_pred_max, first_stage_pred_min);
      } else {
        // assert(false);
        assert(uni_keys.size() != 0);
      }

      // printf(
      //     "[RMI] first stage trained done, start training the second stage\n");
      int aug_keys = 0;
      for (int i = 0; i < key_n; ++i) {
        double index_pred = first_stage->predict(all_values[i].first, model_i);
        unsigned next_stage_model_i = pick_next_stage_model(index_pred);

        second_stage->assign_data(all_values[i].first, all_values[i].second,
                                  next_stage_model_i);

        // real code here
        if (i >= uni_keys.size()) {
          // printf("uni keys sz: %u, all keys sz %u\n",uni_keys.size(),
          // all_keys.size()); assert(false);
        } else {
          this->addrs_map.insert(std::make_pair(uni_keys[i], uni_indexes[i]));
        }

        // my previous key
        if (i - 1 > 0) {
          // augument the previous model
          double index_preda = first_stage->predict(uni_keys[i - 1], 0);
          unsigned next_stage_model_i_1 = pick_next_stage_model(index_preda);

          if (next_stage_model_i_1 != next_stage_model_i) {
            // augument the data
            second_stage->assign_data(uni_keys[i - 1], uni_indexes[i - 1],
                                      next_stage_model_i);
            aug_keys += 1;
          }
          // my key
          second_stage->assign_data(all_values[i].first, all_values[i].second,
                                    next_stage_model_i);
          // my next key
          if (i + 1 < uni_keys.size()) {
            // check next key
            double index_predb = first_stage->predict(uni_keys[i + 1], 0);
            unsigned next_stage_model_i_1_ = pick_next_stage_model(index_predb);
            if (next_stage_model_i_1_ != next_stage_model_i) {
              // augument the data
              second_stage->assign_data(uni_keys[i + 1], uni_indexes[i + 1],
                                        next_stage_model_i);
              aug_keys += 1;
            }
          }
        }
      }
      // printf("[RMI] total %d keys augumented\n", aug_keys);
    }
  }

  /*!
    Add a specific key to a specific model
  */
  void augment_model(const double& key, unsigned model_id) {
    assert(addrs_map.find(key) != addrs_map.end());
    second_stage->assign_data(key, addrs_map[key], model_id);
  }

  void finish_train() {
    if (all_values.empty()) return;
    // prepare 2st stage model with fed in data
    for (int model_i = 0; model_i < second_stage->get_model_n(); ++model_i) {
      // printf("train second stage: %d\n", model_i);
      std::vector<double>& keys = second_stage->data_in[model_i].first;
      std::vector<learned_addr_t>& indexes =
          second_stage->data_in[model_i].second;
      if (keys.size() == 0) {
        // printf("model: %d has 0 training data.\n",model_i);
      }

      // let it track the errors itself
      second_stage->prepare_last(keys, indexes, model_i);
    }
    // printf("second stage done\n");
    if (first_stage->data_in.size() > 0) {
      first_stage->data_in.clear();
    }
    // printf("clear first stage done\n");
    second_stage->data_in.clear();
    all_keys.clear();
    all_values.clear();
    addrs_map.clear();
  }

  void predict_pos(const double key, learned_addr_t& pos) {
    double index_pred = first_stage->predict(key, 0);
    unsigned next_stage_model_i = pick_next_stage_model(index_pred);
    second_stage->predict_last(key, pos,next_stage_model_i);
  }

 public:
  inline unsigned pick_model_for_key(double key) {
    double index_pred = first_stage->predict(key, 0);
    return pick_next_stage_model(index_pred);
  }

  inline unsigned pick_next_stage_model(double index_pred) {
    unsigned next_stage_model_n = second_stage->get_model_n();
    unsigned next_stage_model_i;


    if (index_pred >= key_n) {
      // if(index_pred >= max_addr) {
      next_stage_model_i = next_stage_model_n - 1;
    } else if (index_pred < 0) {
      next_stage_model_i = 0;
    } else {
      next_stage_model_i =
          static_cast<unsigned>(index_pred / key_n * next_stage_model_n);
    }
    return next_stage_model_i;
  }



 private:
  const RMIConfig config;

 public:
  std::vector<double> all_keys;  // not valid after calling finish_insert
  std::vector<std::pair<double, double>> all_values;
  std::vector<learned_addr_t>
      all_addrs;  // not valid after calling finish_insert
  std::map<double, learned_addr_t> addrs_map;  // XD: I add this
  learned_addr_t max_addr = 0;
  bool scale = false;

  unsigned key_n;

  double first_stage_pred_max, first_stage_pred_min;

  LRStage* first_stage;
  LRStage* second_stage;
};
#endif
/*
template <class Weight_T>
class RMIMixTop {
 public:
  RMIMixTop(const RMIConfig& config) : config(config) {
    assert(config.stage_configs.size() == 2);
    assert(config.stage_configs.front().model_n == 1);
    assert(config.stage_configs[0].model_type ==
           RMIConfig::StageConfig::MixTopModel);
    assert(config.stage_configs[1].model_type ==
           RMIConfig::StageConfig::LinearRegression);

    // #ifdef EVENLY_ASSIGN
    //     COUT_THIS("RMI use new-dispatch!");
    // #else
    //     COUT_THIS("RMI use original-dispatch!");
    // #endif
    // init models stage by stage

    first_stage = new MixTopStage<Weight_T>(
        1, 1, config.stage_configs[0].mix_top_config.width,
        config.stage_configs[0].mix_top_config.depth,
        config.stage_configs[0].mix_top_config.weight_dir,
        config.stage_configs[0].mix_top_config.nn_ranges);
    second_stage = new LRStage(config.stage_configs[1].model_n);
  }

  ~RMIMixTop() {
    delete first_stage;
    delete second_stage;
  }

  RMIMixTop(const RMIMixTop&) = delete;
  RMIMixTop(RMIMixTop&) = delete;

  void insert(const double key) { all_keys.push_back(key); }

  void insert_w_idx(const double key, learned_addr_t addr) {
    all_keys.push_back(key);
    all_addrs.push_back(addr);
  }

  void finish_insert() {
    assert(all_keys.size() > 0);

    key_n = all_keys.size();

    sort(all_keys.begin(), all_keys.end());

    // extract <key,index> of unique keys
    double prev_unique_key = all_keys.front();
    std::vector<double> unique_keys{all_keys.front()};
    std::vector<learned_addr_t> unique_indexes{0};

    for (int index = 0; index < all_keys.size(); ++index) {
      double this_key = all_keys[index];
      if (this_key != prev_unique_key) {
        prev_unique_key = this_key;
        unique_keys.push_back(this_key);
        unique_indexes.push_back(index);
      }
    }

    COUT_THIS("number of unique key to RMI: " << unique_keys.size());

    // feed all data to the only model in the 1st stage
    first_stage->reset_data();
    second_stage->reset_data();

    for (int i = 0; i < unique_keys.size(); ++i) {
      first_stage->data_in.front().first.push_back(unique_keys[i]);
      first_stage->data_in.front().second.push_back(unique_indexes[i]);
    }

    // prepare 1st stage model with fed in data
    for (int model_i = 0; model_i < first_stage->get_model_n(); ++model_i) {
      std::vector<double>& uni_keys = first_stage->data_in[model_i].first;
      std::vector<learned_addr_t>& uni_indexes =
          first_stage->data_in[model_i].second;

      first_stage->prepare(uni_keys, uni_indexes, model_i, first_stage_pred_max,
                           first_stage_pred_min);

      for (int i = 0; i < uni_keys.size(); ++i) {
        double index_pred = first_stage->predict(uni_keys[i], model_i);
        unsigned next_stage_model_i = pick_next_stage_model(index_pred);
        second_stage->assign_data(uni_keys[i], uni_indexes[i],
                                  next_stage_model_i);
        assert(false);
        this->addrs_map.insert(std::make_pair(uni_keys[i], uni_indexes[i]));
      }
    }

    // prepare 2st stage model with fed in data
    for (int model_i = 0; model_i < second_stage->get_model_n(); ++model_i) {
      std::vector<double>& keys = second_stage->data_in[model_i].first;
      std::vector<learned_addr_t>& indexes =
          second_stage->data_in[model_i].second;

      // let it track the errors itself
      second_stage->prepare_last(keys, indexes, model_i);
    }

    first_stage->data_in.clear();
    second_stage->data_in.clear();
    all_keys.clear();
    all_addrs.clear();
    addrs_map.clear();
  }

  // private:
  inline unsigned pick_next_stage_model(double index_pred) {
    unsigned next_stage_model_n = second_stage->get_model_n();
    unsigned next_stage_model_i;

#ifdef EVENLY_ASSIGN
    assert(false);
    next_stage_model_i = static_cast<unsigned>(
        (index_pred - first_stage_pred_min) /
        (first_stage_pred_max + 1 - first_stage_pred_min) * next_stage_model_n);
#else
    if (index_pred >= key_n) {
      next_stage_model_i = next_stage_model_n - 1;
    } else if (index_pred < 0) {
      next_stage_model_i = 0;
    } else {
      next_stage_model_i =
          static_cast<unsigned>(index_pred / key_n * next_stage_model_n);
    }
#endif
    return next_stage_model_i;
  }

 private:
  const RMIConfig config;

 public:
  std::vector<double> all_keys;  // not valid after calling finish_insert
  std::vector<learned_addr_t>
      all_addrs;  // not valid after calling finish_insert
  std::map<double, learned_addr_t> addrs_map;
  // learned_addr_t              max_addr = 0;
  // bool scale = false;

  // unsigned key_n;
  learned_addr_t key_n;

  double first_stage_pred_max, first_stage_pred_min;

  MixTopStage<Weight_T>* first_stage;
  LRStage* second_stage;
};
*/

