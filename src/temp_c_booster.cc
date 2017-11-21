/**
 * \file temp_c_booster.cc
 * \brief temporary C lib for replacing python code, to make them faster
 *        mainly for rule-based agents
 */

#include <algorithm>
#include <cmath>
#include <vector>
#include "utility/utility.h"

using magent::utility::NDPointer;

void runaway_infer_action(float *obs_buf, float *feature_buf, int n, int height, int width, int n_channel,
                          int attack_base, int *act_buf, int away_channel, int move_back) {
    int wall  = 0;
    int food  = 1;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        NDPointer<float, 3> obs(obs_buf + i * height*width*n_channel, {{height, width, n_channel}});

        bool found = false;
        for (int row = height-3; row <= height-1 && !found; row++) {
            for (int col = width/2 - 1; col <= width/2 + 1 && !found; col++) {
                if (obs.at(row, col, away_channel) > 0.5) {
                    found = true;
                }
            }
        }
        if (found) {
            act_buf[i] = move_back;
        } else {
            act_buf[i] = move_back + 1;
        }
    }
}

void rush_prey_infer_action(float *obs_buf, float *feature_buf, int n, int height, int width, int n_channel,
                            int *act_buf, int attack_channel, int attack_base,
                            int *view2attack_buf, float threshold) {
    NDPointer<int, 2> view2attack(view2attack_buf, {{height, width}});

    int enemy = attack_channel;
    int wall  = 0;
    int food  = 1;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (feature_buf[i] < threshold) {
            NDPointer<float, 3> obs(obs_buf + i * height*width*n_channel, {{height, width, n_channel}});

            int action = -1;
            bool found = false;
            bool found_attack = false;

            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    if (obs.at(row, col, enemy) > 0.5 || obs.at(row, col, food) > 0.5) {
                        found = true;
                        if (view2attack.at(row, col) != -1) {
                            found_attack = true;
                            action = view2attack.at(row, col);
                            break;
                        }
                    }
                }
                if (found_attack)
                    break;
            }

            if (action == -1) {
                if (found && (int)(obs.at(height-1, width/2, wall) + 0.5) != 1)
                    act_buf[i] = 0;
                else
                    act_buf[i] = (int)(random() % (attack_base));
            } else
                act_buf[i] = attack_base + action;
        } else {
            act_buf[i] = (int)(random() % (attack_base));
        }
    }
}

int get_action(const std::pair<int, int> &disp, bool stride) {
    int action = -1;
    if (disp.first < 0) {
        if (disp.second < 0) {
            action = 1;
        } else if (disp.second == 0) {
            action = stride ? 0 : 2;
        } else {
            action = 3;
        }
    } else if (disp.first == 0) {
        if (disp.second < 0) {
            action = stride ? 4 : 5;
        } else if (disp.second == 0) {
            action = 6;
        } else {
            action = stride ? 8 : 7;
        }
    } else {
        if (disp.second < 0) {
            action = 9;
        } else if (disp.second == 0) {
            action = stride ? 12 : 10;
        } else {
            action = 11;
        }
    }
    return action;
}

void gather_infer_action(float *obs_buf, float *hp_buf, int n, int height, int width, int n_channel,
                         int *act_buf, int attack_base, int *view2attack_buf) {
    NDPointer<int, 2> view2attack(view2attack_buf, {{height, width}});

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        NDPointer<float, 3> obs(obs_buf + i * height*width*n_channel, {{height, width, n_channel}});
        int action = -1;

        if (action == -1) {
            std::vector<int> att_vector;
            std::vector<std::pair<int, int>> vector;

            // find food
            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++) {
                    if (fabs(obs.at(row, col, 4) - 1.0) < 1e-10) {
                        if (view2attack.at(row, col) != -1) {
                            att_vector.push_back(view2attack.at(row, col) + attack_base);
                        } else {
                            int d_row = row - height/2, d_col = col - width/2;
                            if (d_row == d_col && abs(d_col) == 1) {
                                if (rand() & 1)
                                    d_row = 0;
                                else
                                    d_col = 0;
                            }
                            vector.push_back(std::make_pair(d_row, d_col));
                        }
                    }
                }
            if (!att_vector.empty()) {
                action = att_vector[rand() % att_vector.size()];
            } else if (!vector.empty()) {
                action = get_action(vector[0], false);
            }
        }

        // use minimap to navigation
        if (action == -1) {
            std::pair<int, int> mypos = std::make_pair(-1, -1);
            std::vector<std::pair<float, std::pair<int, int>>> vector;

            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++) {
                    if (obs.at(row, col, 3) > 1.0) {
                        mypos = std::make_pair(row, col);
                    }
                }

            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++) {
                    if (obs.at(row, col, 6) > 0.0) {
                        vector.push_back(std::make_pair(obs.at(row, col, 6),
                                                        std::make_pair(row - mypos.first, col - mypos.second)));
                    }
                }
            std::sort(vector.rbegin(), vector.rend());
            action = get_action(vector[rand() % vector.size()].second, true);
            if (action == 6) {
                action = rand() % attack_base;
            }
        }

        act_buf[i] = action;
    }
}
