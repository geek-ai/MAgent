/**
 * \file reward_description.cc
 * \brief implementation of reward description
 */

#include "assert.h"

#include "RewardEngine.h"
#include "GridWorld.h"

namespace magent {
namespace gridworld {

bool AgentSymbol::bind_with_check(void *entity) {
    // bind agent symbol to entity with correctness check
    Agent *agent = (Agent *)entity;
    if (group != agent->get_group())
        return false;
    if (index != -1 && index != agent->get_index())
        return false;
    this->entity = agent;
    return true;
}

/**
 * some interface for python bind
 */
void GridWorld::define_agent_symbol(int no, int group, int index) {
//    LOG(TRACE) << "define agent symbol %d (group=%d index=%d)\n", no, group, index);
    if (no >= agent_symbols.size()) {
        agent_symbols.resize((unsigned)no + 1);
    }
    agent_symbols[no].group = group;
    agent_symbols[no].index = index;
}

void GridWorld::define_event_node(int no, int op, int *inputs, int n_inputs) {
//    TRACE_PRINT("define event node %d op=%d inputs=[", no, op);
//    for (int i = 0; i < n_inputs; i++)
//        TRACE_PRINT("%d, ", inputs[i]);
//    TRACE_PRINT("]\n");
    if (no >= event_nodes.size()) {
        event_nodes.resize((unsigned)no + 1);
    }

    event_nodes[no].op = (EventOp)op; // be careful here
    for (int i = 0; i < n_inputs; i++)
        event_nodes[no].raw_parameter.push_back(inputs[i]);
}

void GridWorld::add_reward_rule(int on, int *receivers, float *values, int n_receiver,
                                bool is_terminal, bool auto_value) {
//    TRACE_PRINT("define rule on=%d rec,val=[", on);
//    for (int i = 0; i < n_receiver; i++)
//        TRACE_PRINT("%d %g, ", receiver[i], value[i]);
//    TRACE_PRINT("]\n");

    RewardRule rule;

    rule.raw_parameter.push_back(on);
    for (int i = 0; i < n_receiver; i++) {
        rule.raw_parameter.push_back(receivers[i]);
        rule.values.push_back(values[i]);
    }
    rule.is_terminal = is_terminal;
    rule.auto_value  = auto_value;

    reward_rules.push_back(rule);
}

void GridWorld::collect_related_symbol(EventNode &node) {
    switch (node.op) {
        // Logic operation
        case OP_AND: case OP_OR:
            collect_related_symbol(*node.node_input[0]);
            collect_related_symbol(*node.node_input[1]);
            // union related symbols
            node.related_symbols.insert(node.node_input[0]->related_symbols.begin(), node.node_input[0]->related_symbols.end());
            node.related_symbols.insert(node.node_input[1]->related_symbols.begin(), node.node_input[1]->related_symbols.end());
            // union infer map
            node.infer_map.insert(node.node_input[0]->infer_map.begin(), node.node_input[0]->infer_map.end());
            node.infer_map.insert(node.node_input[1]->infer_map.begin(), node.node_input[1]->infer_map.end());
            break;
        case OP_NOT:
            collect_related_symbol(*node.node_input[0]);
            // union related symbols
            node.related_symbols.insert(node.node_input[0]->related_symbols.begin(), node.node_input[0]->related_symbols.end());
            // union infer map
            node.infer_map.insert(node.node_input[0]->infer_map.begin(), node.node_input[0]->infer_map.end());
            break;
        // Binary-agent operation
        case OP_KILL: case OP_COLLIDE: case OP_ATTACK:
            node.related_symbols.insert(node.symbol_input[0]);
            node.related_symbols.insert(node.symbol_input[1]);
            break;
        // Unary-agent operation
        case OP_AT: case OP_IN: case OP_DIE: case OP_IN_A_LINE: case OP_ALIGN:
            node.related_symbols.insert(node.symbol_input[0]);
            break;
        default:
            LOG(FATAL) << "invalid event op in GridWorld::collect_related_symbol";
    }
}

void GridWorld::init_reward_description() {
    // from serial data to pointer
    for (int i = 0; i < event_nodes.size(); i++) {
        EventNode &node = event_nodes[i];
        switch (node.op) {
            case OP_AND: case OP_OR:
                node.node_input.push_back(&event_nodes[node.raw_parameter[0]]);
                node.node_input.push_back(&event_nodes[node.raw_parameter[1]]);
                break;
            case OP_NOT:
                node.node_input.push_back(&event_nodes[node.raw_parameter[0]]);
                break;
            case OP_KILL: case OP_COLLIDE: case OP_ATTACK:
                node.symbol_input.push_back(&agent_symbols[node.raw_parameter[0]]);
                node.symbol_input.push_back(&agent_symbols[node.raw_parameter[1]]);
                node.infer_map.insert(std::make_pair(node.symbol_input[0], node.symbol_input[1]));
                break;
            case OP_AT:
                node.symbol_input.push_back(&agent_symbols[node.raw_parameter[0]]);
                node.int_input.push_back(node.raw_parameter[1]);
                node.int_input.push_back(node.raw_parameter[2]);
                break;
            case OP_IN:
                node.symbol_input.push_back(&agent_symbols[node.raw_parameter[0]]);
                node.int_input.push_back(node.raw_parameter[1]);
                node.int_input.push_back(node.raw_parameter[2]);
                node.int_input.push_back(node.raw_parameter[3]);
                node.int_input.push_back(node.raw_parameter[4]);
                break;
            case OP_DIE: case OP_IN_A_LINE: case OP_ALIGN:
                node.symbol_input.push_back(&agent_symbols[node.raw_parameter[0]]);
                break;
            default:
                LOG(FATAL) << "invalid event op in GridWorld::init_reward_description";
        }
    }

    for (int i = 0; i < reward_rules.size(); i++) {
        RewardRule &rule = reward_rules[i];
        rule.on = &event_nodes[rule.raw_parameter[0]];
        for (int j = 1; j < rule.raw_parameter.size(); j++) {
            rule.receivers.push_back(&agent_symbols[rule.raw_parameter[j]]);
        }
    }

    // calc related symbols
    for (int i = 0; i < event_nodes.size(); i++) {
        collect_related_symbol(event_nodes[i]);
    }

    // for every reward rule, find all the input and build inference graph
    for (int i = 0; i < reward_rules.size(); i++) {
        std::vector<AgentSymbol*> input_symbols;
        std::vector<AgentSymbol*> infer_obj;
        std::set<AgentSymbol*> added;

        EventNode &on = *reward_rules[i].on;
        // first pass, scan to find infer pair, add them
        for (auto sub_iter = on.related_symbols.begin(); sub_iter != on.related_symbols.end();
             sub_iter++) {
            if (added.find(*sub_iter) != added.end()) // already be inferred
                continue;

            auto obj_iter = on.infer_map.find(*sub_iter);
            if (obj_iter != on.infer_map.end()) { // can infer object
                input_symbols.push_back(*sub_iter);
                infer_obj.push_back(obj_iter->second);

                added.insert(*sub_iter);
                added.insert(obj_iter->second);
            }
        }

        // second pass, add remaining symbols
        for (auto sub_iter = on.related_symbols.begin(); sub_iter != on.related_symbols.end();
             sub_iter++) {
            if (added.find(*sub_iter) == added.end()) {
                input_symbols.push_back(*sub_iter);
                infer_obj.push_back(nullptr);
            }
        }

        reward_rules[i].input_symbols = input_symbols;
        reward_rules[i].infer_obj     = infer_obj;
    }

    /**
     * semantic check
     * 1. the object of attack, collide, kill cannot be a group
     * 2. any non-deterministic receiver must be involved in the triggering event
     */
    for (int i = 0; i < reward_rules.size(); i++) {
        // TODO: omitted temporally
    }

    // print rules for debug
    /*for (int i = 0; i < reward_rules.size(); i++) {
        printf("on: %d\n", (int)reward_rules[i].on->op);
        printf("input symbols: ");
        for (int j = 0; j < reward_rules[i].input_symbols.size(); j++) {
            printf("(%d,%d) ", reward_rules[i].input_symbols[j]->group,
                               reward_rules[i].input_symbols[j]->index);
            if (reward_rules[i].infer_obj[j] != nullptr) {
                printf("-> (%d,%d)  ", reward_rules[i].infer_obj[j]->group,
                                     reward_rules[i].infer_obj[j]->index);
            }
        }
        printf("\n");
    }*/
}

bool GridWorld::calc_event_node(EventNode *node, RewardRule &rule) {
    bool ret;
    switch (node->op) {
        case OP_ATTACK: case OP_KILL: case OP_COLLIDE: {
            Agent *sub, *obj;
            // object must be an agent, cannot be a group !
            assert(!node->symbol_input[1]->is_all());
            obj = (Agent *)node->symbol_input[1]->entity;
            if (node->symbol_input[0]->is_all()) {
                const std::vector<Agent*> &agents = groups[node->symbol_input[0]->group].get_agents();
                ret = true;
                for (int i = 0; i < agents.size(); i++) {
                    sub = agents[i];

                    if (!(sub->get_last_op() == node->op && sub->get_op_obj() == obj)) {
                        ret = false;
                        break;
                    }
                }
            } else {
                sub = (Agent *)node->symbol_input[0]->entity;
                ret = sub->get_last_op() == node->op && sub->get_op_obj() == obj;
            }
        }
            break;
        case OP_ALIGN: {
            // subject must be an agent, cannot be a group!
            assert(!node->symbol_input[0]->is_all());

            Agent *sub = (Agent *)node->symbol_input[0]->entity;

            // int align = map.get_align(sub);
            Position pos = sub->get_pos();
            assert(pos.x < width && pos.y < height);
            int align = counter_x[pos.x] + counter_y[pos.y];

            if (rule.auto_value) {
                assert(rule.values.size() == 1);
                rule.values[0] = align - 1;
                ret = true;
            } else {
                ret = align > 1;
            }
        }
            break;
        case OP_IN_A_LINE: {
            assert(node->symbol_input[0]->is_all());
            std::vector<Agent*> &agents = groups[node->symbol_input[0]->group].get_agents();
            if (agents.size() < 2) {
                ret = true;
            } else {
                ret = false;
                // check if they are in a line, condition : 1. x is the same  2. max_y - min_y + 1 = #agent
                int dx, dy;
                dx = agents[0]->get_pos().x - agents[1]->get_pos().x;
                dy = agents[0]->get_pos().y - agents[1]->get_pos().y;
                bool in_line = true;
                if (dx == 0 && dy != 0) {
                    int min_y, max_y;
                    int base_x = agents[0]->get_pos().x;
                    min_y = max_y = agents[0]->get_pos().y;
                    for (int i = 1; i < agents.size() && in_line; i++) {
                        Position pos = agents[i]->get_pos();
                        min_y = std::min(pos.y, min_y); max_y = std::max(pos.y, max_y);
                        in_line = (base_x == pos.x);
                    }
                    ret = in_line && max_y - min_y + 1 == agents.size();
                } else if (dx != 0 && dy == 0) {
                    int min_x, max_x;
                    int base_y = agents[0]->get_pos().y;
                    min_x = max_x = agents[0]->get_pos().x;
                    for (int i = 1; i < agents.size() && in_line; i++) {
                        Position pos = agents[i]->get_pos();
                        min_x = std::min(pos.x, min_x); max_x = std::max(pos.x, max_x);
                        in_line = (base_y == pos.y);
                    }
                    ret = in_line && max_x - min_x + 1 == agents.size();
                }
            }
        }
            break;
        case OP_AT: {
            std::vector<int> &int_input = node->int_input;
            if (node->symbol_input[0]->is_all()) {
                const std::vector<Agent*> &agents = groups[node->symbol_input[0]->group].get_agents();
                ret = true;
                for (int i = 0; i < agents.size(); i++) {
                    Position pos = agents[i]->get_pos();
                    if (!(pos.x == int_input[0] && pos.y == int_input[1])) {
                        ret = false;
                        break;
                    }
                }
            } else {
                Agent *sub = (Agent *)node->symbol_input[0]->entity;
                Position pos = sub->get_pos();
                ret = (pos.x == int_input[0] && pos.y == int_input[1]);
            }
        }
            break;
        case OP_IN: {
            std::vector<int> &int_input = node->int_input;
            if (node->symbol_input[0]->is_all()) {
                const std::vector<Agent*> &agents = groups[node->symbol_input[0]->group].get_agents();
                ret = true;
                for (int i = 0; i < agents.size(); i++) {
                    Position pos = agents[i]->get_pos();
                    if (!((pos.x > int_input[0] && pos.x < int_input[2]
                        && pos.y > int_input[1] && pos.y < int_input[3]))) {
                        ret = false;
                        break;
                    }
                }
            } else {
                Agent *sub = (Agent *)node->symbol_input[0]->entity;
                Position pos = sub->get_pos();
                ret = ((pos.x > int_input[0] && pos.x < int_input[2]
                    && pos.y > int_input[1] && pos.y < int_input[3]));
            }
        }
            break;
        case OP_DIE: {
            if (node->symbol_input[0]->is_all()) {
                const std::vector<Agent*> &agents = groups[node->symbol_input[0]->group].get_agents();
                ret = true;
                for (int i = 0; i < agents.size(); i++) {
                    if (!agents[i]->is_dead()) {
                        ret = false;
                        break;
                    }
                }
            } else {
                Agent *sub = (Agent *)node->symbol_input[0]->entity;
                ret = sub->is_dead();
            }
        }
            break;

        case OP_AND:
            ret = calc_event_node(node->node_input[0], rule)
                && calc_event_node(node->node_input[1], rule);
            break;
        case OP_OR:
            ret = calc_event_node(node->node_input[0], rule)
                || calc_event_node(node->node_input[1], rule);
            break;
        case OP_NOT:
            ret = !calc_event_node(node->node_input[0], rule);
            break;

        default:
            LOG(FATAL) << "invalid op of EventNode in GridWorld::calc_event_node";
    }

    return ret;
}

void GridWorld::calc_rule(std::vector<AgentSymbol *> &input_symbols,
                          std::vector<AgentSymbol *> &infer_obj,
                          RewardRule &rule, int now) {
    if (now == input_symbols.size()) { // DFS last layer
        if (calc_event_node(rule.on, rule)) { // if it is true, assign reward
            rule.trigger = true;
            const std::vector<AgentSymbol*> &receivers = rule.receivers;
            for (int i = 0; i < receivers.size(); i++) {
                AgentSymbol *sym = receivers[i];
                if (sym->is_all()) {
                    groups[sym->group].add_reward(rule.values[i]);
                } else {
                    Agent* agent = (Agent *)sym->entity;
                    agent->add_reward(rule.values[i]);
                }
            }
        }
    } else { // scan every possible permutation
        AgentSymbol *sym = input_symbols[now];
        if (sym->is_any()) {
            const std::vector<Agent*> &agents = groups[sym->group].get_agents();
            int base = groups[sym->group].get_recursive_base() + 1;
            base = 0;

            for (int i = base; i < agents.size(); i++) {
                groups[sym->group].set_recursive_base(i);
                sym->entity = (void *)agents[i];

                if (agents[i]->get_involved())
                    continue;
                agents[i]->set_involved(true);

                if (infer_obj[now] != nullptr) {   // can infer in this step
                    void *entity = agents[i]->get_op_obj();
                    if (entity != nullptr && infer_obj[now]->bind_with_check(entity)) {
                        calc_rule(input_symbols, infer_obj, rule, now + 1);
                    }
                } else {
                    calc_rule(input_symbols, infer_obj, rule, now + 1);
                }
                agents[i]->set_involved(false);
            }
        } else if (sym->is_all()) {
            if (infer_obj[now] != nullptr) {
                Group &g = groups[sym->group];
                if (g.get_agents().size() > 0)  {
                    void *entity = g.get_agents()[0]->get_op_obj(); // pick first agent to infer
                    if (entity != nullptr && infer_obj[now]->bind_with_check(entity)) {
                        calc_rule(input_symbols, infer_obj, rule, now + 1);
                    }
                }
            } else {
                calc_rule(input_symbols, infer_obj, rule, now + 1);
            }
        } else { // deterministic ID
            Group &g = groups[sym->group];
            if (sym->index < g.get_size()) {
                Agent *agent = g.get_agents()[sym->index];
                sym->entity = (void *)agent;

                if (infer_obj[now] != nullptr) {
                    if (agent->get_op_obj() != nullptr) {
                        if (infer_obj[now]->bind_with_check(agent->get_op_obj())) {
                            calc_rule(input_symbols, infer_obj, rule, now + 1);
                        }
                    }
                }
            }
        }
    }
}



} // namespace gridworld
} // namespace magent