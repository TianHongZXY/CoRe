import jsonlines
import math
import time
import logging
import coloredlogs
from collections import defaultdict
from itertools import accumulate
import heapq
import gc
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
from sample_sequence import top_k_logits
import copy
from calculator import use_calculator
from math_data_model import extract_answer, is_correct
logger = logging.getLogger('__file__')
coloredlogs.install(level='INFO', logger=logger)


def pushq(listy, capacity, item):
    """item: (score, text)"""
    if item in listy:
        print("repeat item: ", item)
    #      return

    if len(listy) < capacity:
        heapq.heappush(listy, item)
    else:
        heapq.heappushpop(listy, item)


class State():
    """
    This class represents the state of a node.
    param num_gen: length of tokens to be allowed to generate
    param is_terminal: whether a leaf node
    param token_ids: the token ids
    """

    def __init__(self, num_gen, token_ids, prob=1.0):
        self.num_gen = num_gen
        self.is_terminal = (self.num_gen == 0)
        self.token_ids = token_ids
        self.prob = prob

    def next(self, next_token_ids, prob):
        if self.num_gen <= 0:
            raise ValueError("exceed maximal allowed length")
        return State(self.num_gen-1, next_token_ids, prob)

    def __hash__(self):
        return self.token_ids[0][0]

    def __eq__(self, others):
        return self.__hash__() == others.__hash__()


class Node(object):
    """
    This class defines the node of a search tree
    param visit: number of visit
    param parent: parent node
    param: state: state of current node
    param next_token_probs: prob. distributiion of next token
    param *mems: additional params in case of recomputing
    """

    def __init__(self, parent, state, next_token_probs, max_num_children, mems):
        self.visit = 0
        self.reward = 0.0
        self.parent = parent
        if self.parent is None:
            self.emitted_tokens = defaultdict(lambda:1)
        else:
            self.emitted_tokens = parent.emitted_tokens.copy()
            token_ids = state.token_ids.view(-1).tolist()
            for token in token_ids:
                self.emitted_tokens[token] += 1

        self.state = state
        self.children = []
        self.max_num_children = max_num_children
        self.next_token_probs = next_token_probs
        self.mems = mems

    def __repr__(self):
        return f"token: {tokenizer.decode(self.state.token_ids.view(-1))}, visit: {self.visit}, reward: {self.reward}, prob: {self.state.prob}"

    def add_child(self, child_state, child_next_token_probs, max_num_children, child_mems):
        child_node = Node(self, child_state, child_next_token_probs, max_num_children, child_mems)
        self.children.append(child_node)

    def update(self, reward, decay_rate = 0.95):
        self.visit += 1
        self.reward += decay_rate*reward

    def empty_cache(self):
        self.mems = None
        self.next_token_probs = None
        self.emitted_tokens = None
        torch.cuda.empty_cache()
        gc.collect()

    def is_fully_expanded(self):
        return len(self.children) == self.max_num_children

def add_common_ctrl_args(parser):
    "generation control"
    group = parser.add_argument_group('commn ctrl', 'configurations')

    group.add_argument("--device", type=str, default='cuda')
    group.add_argument("--temperature", type=float, default=1.1, help='sampling temperature')
    group.add_argument("--top_p", type=float, default=0.0)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument("--seed", type=int, default=19990303)

    return parser

def add_mcts_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('mcts', 'configurations')

    group.add_argument("--max_num_children", type=int, default=4, help='maximal number of children of non-root node')
    group.add_argument("--root_max_num_children", type=int, default=10, help='maximal number of children of root node')
    group.add_argument("--roll_out_size", type=int, default=200, help='size of roll out.')
    group.add_argument("--sampling_size", type=int, default=1024, help='maximal sequence length allowed for sampling')
    group.add_argument("--max_length", type=int, default=1024, help='maximal depth of the tree')
    group.add_argument("--max_iter", type=int, default=500, help='maximally allowed iterations')
    group.add_argument("--time_out", type=float, default=180, help='maximally allowed runing time for each sample, in unit of second')
    group.add_argument("--alpha", type=float, default=1.0, help='balance similarity and fluency. Its value is within the range of [0, 1], `1` means ignore fluency while `0` means ignore similarity')
    group.add_argument("--c_out", type=float, default=10.0, help='coeffecient balacing exploration and expolitation')
    group.add_argument("--bp_decay_rate", type=float, default=1.0, help='decay rate during back propagation')
    group.add_argument("--select_strategy", type=MCTS.SELECT_STRATEGY, default=MCTS.SELECT_STRATEGY.BEST, help='select strategy')
    group.add_argument("--annealing_rate", type=float, default=10.0, help='annealing rate. mandatory if `select_strategy` is `MCTS.SELECT_STRATEGY.ANNEALING`')
    group.add_argument("--initialize_tree", dest='initialize_tree', action='store_true')
    group.add_argument("--sample_capacity", type=int, default=100)
    group.add_argument("--expand_length", type=int, default=1)
    group.add_argument("--split", type=str)
    group.add_argument("--expand_repeat_penalty", type=float, default=1.2)
    group.add_argument("--expand_reward_alpha", type=float, default=1.0)

    return parser


def add_gsm8k_args(parser):
    """GSM8k arguments."""

    group = parser.add_argument_group('gsm8k', 'configurations')

    group.add_argument("--verifier_type", type=str)
    group.add_argument("--verifier_name", type=str)
    group.add_argument("--expand_verifier_type", type=str)
    group.add_argument("--expand_verifier_name", type=str)
    group.add_argument("--model_name", type=str)
    group.add_argument("--data", type=str)
    group.add_argument("--timestamp", type=str)
    group.add_argument("--data_name", default="", type=str)

    return parser


class MCTS():
    class SELECT_STRATEGY:
        RANDOM = 0
        ANNEALING = 1
        BEST = 2

    def __init__(self, model, tokenizer, args, device, verifier_model, verifier_head, verifier_tokenizer, expand_verifier_model, expand_verifier_head, expand_verifier_tokenizer, input_token_ids=None, scalar=1.0, label=None, root=None):
        self.model = model
        self.tokenizer = tokenizer
        self.thought_idx = tokenizer.convert_tokens_to_ids("[THOUGHT]")
        self.verifier_model = verifier_model
        self.verifier_head = verifier_head
        self.verifier_tokenizer = verifier_tokenizer
        self.verifier_idx = verifier_tokenizer.convert_tokens_to_ids("[VERIFIER]")
        self.expand_verifier_model = expand_verifier_model
        self.expand_verifier_head = expand_verifier_head
        self.expand_verifier_tokenizer = expand_verifier_tokenizer
        self.expand_verifier_idx = expand_verifier_tokenizer.convert_tokens_to_ids("[VERIFIER]")
        self.step_token_idx = tokenizer('\n').input_ids[0]
        self.args = args
        self.device = device
        self.input_token_ids = input_token_ids

        assert input_token_ids is not None
        self.org_context_length = input_token_ids.size(1)
        self.max_num_gen = self.args.max_length - self.org_context_length
        self.eos_token = self.tokenizer.eos_token
        if args.sample_capacity < 0:
            self.sample_capacity = 2 * self.args.num_per_sample
        else:
            self.sample_capacity = args.sample_capacity
        self.scalar = scalar
        self.label = label
        self.good_cases = []
        self.node_mem_len = 3

        if root is None:
            #  bos_token_id = self.tokenizer(["[QUES]"], return_tensors="pt").to(self.device).input_ids
            next_token_probs, mems = self.get_token_probs(index=0, token_ids=input_token_ids, node=None)
            self.root = Node(parent=None,
                             state=State(num_gen=self.max_num_gen, token_ids=input_token_ids.view(1, -1).to(self.device), prob=1.),
                             next_token_probs=next_token_probs,
                             max_num_children=args.root_max_num_children,
                             mems=mems,
                             )
        else:
            self.root = root

    def search(self):
        if self.args.initialize_tree:
            logger.info('-'*20+'iter. 0' + '-'*20)
            self.initialize_tree_with_input()

        tic = time.time()
        for i in range(self.args.max_iter):
            #  logger.info('-'*20+f'iter. {i+1:4d}'+'-'*20)
            #  if time.time() - tic > self.args.time_out:
            #      break
            #      print('-'*20+f'iter. {i+1:4d}'+'-'*20)
            #      self.printTree()
            #      print('-'*20 + f'time out: {self.args.time_out:4.0f} s' + '-'*20)
            #      return
            self.search_once()
        print('-'*30 + f'SEARCH BEGIN' + '-'*30)
        self.printTree()
        print('-'*20 + f'maximal iterations reached: max_iter = {self.args.max_iter:4d}' + '-'*20)
        print('-'*20 + f'SEARCH END, TIME SPENT: {time.time() - tic} seconds' + '-'*20)

    def search_once(self):
        front = self.search_policy(self.root)
        path_tokens, ippl = self.roll_out(front, self.args.roll_out_size)
        reward = self.reward(path_tokens, ippl)
        self.back_prop(front, reward)
        #  print("New Node: ", front)
        gc.collect()
        torch.cuda.empty_cache()

    def initialize_tree_with_input(self):
        node = self.root
        for token_ids in self.input_token_ids:
            next_token = torch.LongTensor([[token_ids]]).to(self.device)
            child_state = node.state.next(next_token, (node.next_token_probs[0, next_token]).view(-1).mean().detach().cpu().tolist())
            # next_token_probs, *mems = self.get_token_probs(self.max_num_gen - child_state.num_gen, child_state.token_ids, node.emitted_tokens, *node.mems)
            next_token_probs, mems = self.get_token_probs(self.max_num_gen - child_state.num_gen, child_state.token_ids, node)
            if self.max_num_gen - child_state.num_gen > self.node_mem_len:
                mems = None
            node.add_child(child_state, next_token_probs, self.args.max_num_children, mems)
            node = node.children[-1]

        print("-" * 20 + "Whole Tree" + "-" * 20)
        self.printTree()
        reward = self.reward(*self.roll_out(node, self.args.roll_out_size))
        self.back_prop(node, reward)

    def search_policy(self, node):
        # a hack to force 'exploitation'
        logger.debug("enter search_policy")
        last_token_id = node.state.token_ids.view(-1).tolist()
        # while (node.state.is_terminal is False) and (last_token[0] != self.args.eos_token) and ('”' not in eos_token):
        while (node.state.is_terminal is False) and (self.tokenizer.eos_token_id not in last_token_id):
            if len(node.children) == 0:
                return self.expand_multi_step_with_calculator(node)
            elif random.uniform(0, 1) < .5 or node.is_fully_expanded():
                node = self.select(node, self.args.select_strategy) 
            else:
                return self.expand_multi_step_with_calculator(node)
            last_token_id = node.state.token_ids.view(-1).tolist()

        logger.debug("leave search_policy")
        return node

    def expand_multi_step_with_calculator(self, node):
        logger.debug("enter expand multi step with calculator")
        already_tried = [c.state for c in node.children]
        next_token_probs = node.next_token_probs
        next_node_token_ids = []
        next_token_prob = []
        teriminal = False
        for i in range(1, self.args.expand_length + 1):
            next_token = torch.multinomial(next_token_probs, num_samples=1)

            #  multi step expand
            if i == 1:
                tmp_state = node.state.next(next_token, 0)
                if tmp_state in already_tried and tmp_state.is_terminal is False:
                    print(f'Expand repeat penalty triggered! Sampled token: {self.tokenizer.convert_ids_to_tokens([next_token])}, prob: {next_token_probs[0, next_token].item()}')
                    next_token_probs[0, next_token] /= self.args.expand_repeat_penalty # penalty probs of states in already_tried
                    try:
                        next_token = torch.multinomial(next_token_probs, num_samples=1)
                        tmp_state = node.state.next(next_token, 0)
                    except: # no available candidates
                        print("No available candidates due to expand repeat penalty, early stop!")
                        node.max_num_children = len(node.children)
                        return node.children[already_tried.index(tmp_state)]

            prob = next_token_probs[0, next_token].detach().cpu().item()

            if next_token.item() == self.tokenizer.eos_token_id: # or next_token.item() == self.step_token_idx:
                teriminal = (next_token.item() == self.tokenizer.eos_token_id)
                next_node_token_ids.extend(next_token.detach().cpu().view(-1).tolist())
                next_token_prob.append(prob)
                if not teriminal:
                    next_token_probs, mems = self.get_token_probs(-1, torch.tensor(next_node_token_ids, device=self.device).view(1, -1), node)
                else:
                    mems = None
                    next_token_probs = None
                break

            if next_token.item() in EQUALS_TOKENS:
                cur = node
                token_ids = cur.state.token_ids.detach().cpu().view(-1).tolist() + next_node_token_ids + next_token.detach().cpu().view(-1).tolist()
                while cur.parent is not None:
                    #  token_ids = torch.cat((cur.parent.state.token_ids, token_ids), dim=-1)
                    token_ids = cur.parent.state.token_ids.detach().cpu().view(-1).tolist() + token_ids
                    cur = cur.parent
                text = tokenizer.decode(token_ids)
                answer = use_calculator(text)
                if answer is not None:
                    text = text + str(answer) + ">>"
                    next_token = torch.cat((next_token, tokenizer([str(answer) + ">>"], return_tensors="pt").to(self.device).input_ids), dim=-1)
                #  else:
                    #  logger.error(f"= generated but no answer is got, text: {text}, token_ids: {token_ids}.")


            next_node_token_ids.extend(next_token.detach().cpu().view(-1).tolist())
            next_token_prob.append(prob)

            next_token_probs, mems = self.get_token_probs(-1, torch.tensor(next_node_token_ids, device=self.device).view(1, -1), node)


        child_state = State(node.state.num_gen - len(next_node_token_ids), torch.tensor(next_node_token_ids, device=self.device).view(1, -1), np.mean(next_token_prob))
        child_state.is_terminal = teriminal or child_state.is_terminal

        if node.state.num_gen - child_state.num_gen > self.node_mem_len:
            mems = None
        node.add_child(child_state, next_token_probs, self.args.max_num_children, mems)

        logger.debug("leave expand")

        # -----------------calculate expand reward---------------------
        cur = node.children[-1]
        logsum = [math.log(prob+1.0e-20) for prob in next_token_prob]
        ippl = math.exp(sum(logsum) / len(logsum)) # inverse perplexity
        path_tokens = next_node_token_ids[:]
        while cur.parent is not None:
            node_token = cur.parent.state.token_ids.detach().cpu().view(-1).tolist()
            path_tokens = node_token + path_tokens
            cur = cur.parent
        reward = self.expand_reward(path_tokens, ippl)
        reward *= self.args.expand_reward_alpha
        self.back_prop(node.children[-1], reward)
        gc.collect()
        torch.cuda.empty_cache()
        # -------------------------end---------------------------------
        return node.children[-1]

    def select(self, node, select_strategy=SELECT_STRATEGY.ANNEALING):
        logger.debug("enter select")

        best_node = None
        if select_strategy == MCTS.SELECT_STRATEGY.BEST:
            best_score = float("-inf")
            best_nodes = []

            for c in node.children:
                score, _, _ = self.calc_ee_score(c, node)
                if score == best_score:
                    best_nodes.append(c)
                elif score > best_score:
                    best_nodes = [c]
                    best_score = score

            best_node = random.choice(best_nodes)
        elif select_strategy == MCTS.SELECT_STRATEGY.ANNEALING:
            score = []
            for c in node.children:
                score.append(self.calc_ee_score(c, node)[0])

            score = torch.tensor(score)
            probs = torch.softmax(score*self.args.annealing_rate*(self.max_num_gen - node.state.num_gen + 1), dim=0) # quite flat at first few nodes, then more focusing
            idx = torch.multinomial(probs, 1)[0].tolist()
            best_node = node.children[idx]
        else:
            score = []
            for c in node.children:
                score.append(self.calc_ee_score(c, node)[0])
                probs = np.array(score)
                probs /= np.sum(probs)

            idx = list(np.random.multinomial(1, probs)).index(1)
            best_node = node.children[idx]

        logger.debug("leave select")
        return best_node

#      def random_select(self, node):
#          score = []
#          for c in node.children:
#              score.append(self.calc_ee_score(c, node)[0])
#          score = np.array(score)
#          score /= score.sum()
#  #         idx = np.random.choice(len(node.children), 1, p=score)[0]
#          idx = list(np.random.multinomial(1, score)).index(1)
#          return node.children[idx]

    def roll_out(self, node, roll_out_size):
        logger.debug("enter roll_out")

        path_tokens = []
        logsum = []
        def update_tokens(tokens, prob, num_tokens, append=True):
            for i, t in enumerate(tokens):
                num_tokens += 1
                if append:
                    path_tokens.append(t)
                    if i == 0:
                        logsum.append(math.log(prob+1.0e-20))
                    else:
                        logsum.append(0.0)
                else:
                    path_tokens.insert(i, t)
                    if i == 0:
                        logsum.insert(0, math.log(prob+1.0e-20))
                    else:
                        logsum.insert(i, 0.0)
            return num_tokens

        num_tokens = 0
        node_token = node.state.token_ids.detach().cpu().view(-1).tolist()
        num_tokens = update_tokens(node_token, node.state.prob, num_tokens)

        # history tokens
        cur = node
        while cur.parent is not None:
            node_token = cur.parent.state.token_ids.detach().cpu().view(-1).tolist()
            num_tokens = update_tokens(node_token, cur.parent.state.prob, num_tokens, append=False)
            # path_tokens.insert(0, cur.parent.state.token_ids.detach().cpu().view(-1).tolist()[0])
            # if cur.parent.state.prob > 1.0e-20:
            #     logsum.insert(0, math.log(cur.parent.state.prob))
            #     num_tokens += 1
            cur = cur.parent

        cur = node
        while not hasattr(cur, 'mems'):
        #  while cur.mems is None:
            # cur = cur.children[-1]
            # path_tokens.extend(cur.state.token_ids.detach().cpu().view(-1).tolist())
            cur = cur.children[-1]
            node_token = cur.state.token_ids.detach().cpu().view(-1).tolist()
            num_tokens = update_tokens(node_token, cur.state.prob, num_tokens)

        # mems = cur.mems
        # state = cur.state
        # emitted_tokens = cur.emitted_tokens.copy()
        tmp_node = copy.deepcopy(cur)

        tmp_node_token_ids = tmp_node.state.token_ids.view(-1).tolist()
        gen_len = 0
        # while (tmp_node.state.is_terminal is False) and (self.args.eod_token not in eod) and ('”' not in eod_token) and (gen_len < roll_out_size):
        while (tmp_node.state.is_terminal is False) and (self.tokenizer.eos_token_id not in tmp_node_token_ids) and (gen_len < roll_out_size):
            gen_len += 1
            index = self.max_num_gen - tmp_node.state.num_gen
            # probs, *mems = self.get_token_probs(index, state.token_ids, emitted_tokens, *mems)
            probs, mems = self.get_token_probs(index, tmp_node.state.token_ids, tmp_node)
            tmp_node.mems = mems
            next_token = torch.multinomial(probs, num_samples=1)
            prob = (probs[0, next_token]).view(-1).mean().detach().cpu().tolist()

            if next_token.item() in EQUALS_TOKENS:
                #  token_ids = torch.cat((tmp_node.state.token_ids, next_token), dim=-1).view(-1)
                token_ids = path_tokens + [next_token.detach().cpu().item()]
                token_ids = list(filter(lambda x: x is not None, token_ids))
                assert None not in token_ids, f"{token_ids}"
                try:
                    text = tokenizer.decode(token_ids)
                except:
                    logger.error("decode bugs")
                    print(tmp_node)
                    self.printTree()
                    print("length of token_ids: ", len(token_ids), token_ids)
                    raise ValueError()
                answer = use_calculator(text)
                if answer is not None:
                    text = text + str(answer) + ">>"
                    next_token = torch.cat((next_token, tokenizer([str(answer) + ">>"], return_tensors="pt").to(self.device).input_ids), dim=-1)

            tmp_node.state = tmp_node.state.next(next_token, prob)
            tmp_node_token_ids = tmp_node.state.token_ids.view(-1).tolist()

            node_token = next_token.detach().cpu().view(-1).tolist()
            num_tokens = update_tokens(node_token, tmp_node.state.prob, num_tokens)
            # path_tokens.extend(next_token.detach().cpu().view(-1).tolist())
            # logsum.append(math.log(state.prob))
            # num_tokens += 1

            # update frequencies of emitted tokens 
            #  new_token_ids = next_token.view(-1).tolist()
            #  for new_token in new_token_ids:
            #      tmp_node.emitted_tokens[new_token] += 1

        #  cumsum = list(accumulate(logsum))

        #  ippl = math.exp(sum(logsum[burned_len:])/float(num_tokens - burned_len)) # inverse perplexity
        #  path_tokens = path_tokens[burned_len:]
        #  logsum = logsum[burned_len:]
        ippl = math.exp(sum(logsum) / num_tokens) # inverse perplexity

        logger.debug("leave roll_out")
        #  print("path tokens:", self.tokenizer.decode(path_tokens))

        return path_tokens, ippl

    def back_prop(self, node, reward):
        logger.debug("enter back_prop")

        cur = node
        decay_rate = self.args.bp_decay_rate
        while cur is not None:
            cur.update(reward, decay_rate)
            cur = cur.parent
            decay_rate *= decay_rate
        logger.debug("leave back_prop")

    def expand_reward(self, tokens, fluency=1.0):
        output_ids = tokens
        output_text = self.tokenizer.decode(output_ids)
        output_text = output_text.replace(" [ANS] ", "[ANS]")
        q_and_t = output_text.split("[THOUGHT]", maxsplit=1)
        ques, thought = q_and_t[0], "[THOUGHT]" + q_and_t[1]
        if self.args.expand_verifier_type == "bert" or self.args.expand_verifier_type == "deberta":
            inputs_encoding = self.expand_verifier_tokenizer([ques], [thought], return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512).to(self.device)
        elif self.args.expand_verifier_type == "gpt":
            inputs_encoding = self.expand_verifier_tokenizer(ques + thought, return_tensors="pt", add_special_tokens=False).to(self.device)

        verifier_score = self.calc_expand_verifier_score(**inputs_encoding)

        if self.args.alpha < 1.0:
            score = math.pow(verifier_score, self.args.alpha) * math.pow(fluency, 1.0 - self.args.alpha) # strenthen languate model if alpha < 1
        else:
            score = verifier_score

        return score

    def reward(self, tokens, fluency=1.0):
        output_ids = tokens
        output_text = self.tokenizer.decode(output_ids)
        if self.tokenizer.eos_token_id not in output_ids:
            logger.warning(f"No eos token! {output_text}")
        #  output_text = output_text.split(self.eos_token)[0]
        output_text = output_text.replace(" [ANS] ", "[ANS]")
        q_and_t = output_text.split("[THOUGHT]", maxsplit=1)
        ques, thought = q_and_t[0], "[THOUGHT]" + q_and_t[1]
        if self.args.verifier_type == "bert" or self.args.verifier_type == "deberta":
            inputs_encoding = self.verifier_tokenizer([ques], [thought], return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512).to(self.device)
        elif self.args.verifier_type == "gpt":
            inputs_encoding = self.verifier_tokenizer(ques + thought, return_tensors="pt", add_special_tokens=False).to(self.device)

        verifier_score = self.calc_verifier_score(**inputs_encoding)


        if self.args.alpha < 1.0:
            score = math.pow(verifier_score, self.args.alpha) * math.pow(fluency, 1.0 - self.args.alpha) # strenthen languate model if alpha < 1
        else:
            score = verifier_score

        pushq(self.good_cases, self.sample_capacity, (score, thought, verifier_score, fluency))
        #  if verifier_score > 0.5:
            #  print(f'Good Case: verifier score={verifier_score}, fluency={fluency}, final_score={score}\n{output_text}', flush=True)
            #  pushq(self.good_cases, self.sample_capacity, (score, output_text))
        #  else:
            #  print(f'Bad Case: verifier score={verifier_score}, fluency={fluency}, final_score={score}\n{output_text}', flush=True)

        return score

    def calc_expand_verifier_score(self, **inputs_encoding):
        output = self.expand_verifier_model(**inputs_encoding)
        # Bert select the first token(cls)，GPT select the last token
        verifier_logits = output.logits[:, 0 if self.args.expand_verifier_type == "bert" or self.args.expand_verifier_type == "deberta" else -1, self.expand_verifier_idx].half()  # Expected shape = (bs, )
        verifier_predictions = self.expand_verifier_head(verifier_logits.unsqueeze(-1))  # Expected shape = ()

        return verifier_predictions.item()

    def calc_verifier_score(self, **inputs_encoding):
        output = self.verifier_model(**inputs_encoding)
        verifier_logits = output.logits[:, 0 if self.args.verifier_type == "bert" or self.args.verifier_type == "deberta" else -1, self.verifier_idx].half()  # Expected shape = (bs, )
        verifier_predictions = self.verifier_head(verifier_logits.unsqueeze(-1))  # Expected shape = ()

        return verifier_predictions.item()

    def calc_ee_score(self, node, parent):
        exploit = node.reward / max(node.visit, 1)
        explore = math.sqrt(float(parent.visit)) / float(1 + node.visit)

        explore = self.scalar * explore * node.state.prob
        return exploit + explore, exploit, explore

    def printTree(self, sep=8, start_node=None):
        def _dfs(parent, level=0):
            if level > 0:
                strs = ''
                for i in range(level-1):
                    strs += '|' + ' '*sep
                strs += '|->'
                #  token = self.tokenizer.convert_ids_to_tokens(parent.state.token_ids.view(-1).detach().cpu().tolist())
                token = self.tokenizer.decode(parent.state.token_ids.view(-1).detach().cpu().tolist())
                score, exploit, explore = self.calc_ee_score(parent, parent.parent)
                #  print(f'{strs}{token}(score:{score:.2f},exploit:{exploit:.2f},explore:{explore:.2f})')
                return_symbol = "\n"
                return_symbol_str = "\\n"
                print(f'{strs}{token.replace(return_symbol, return_symbol_str)}({score:.2f},{exploit:.2f},{explore:.2f})', flush=True)
            for node in parent.children:
                _dfs(node, level + 1)

        if start_node is None:
            _dfs(self.root)
        else:
            _dfs(start_node)

    def traverse(self):
        def recursive(node, tokens, total):
            sent = tokens.copy()
            sent.append(node.state.token_ids.view(1).detach().cpu().tolist()[0])

            if node.state.is_terminal or len(node.children) == 0:
                total.append(self.tokenizer.convert_tokens_to_ids(sent[1:]))
                return
            for c in node.children:
                recursive(c, sent, total)

        full_path = []
        recursive(self.root, [], full_path)
        return full_path

    def get_token_probs(self, index, token_ids, node):
        self.model.eval()
        #  inputs_encoding = self.tokenizer(text, return_attention_mask=True, return_tensors="pt", padding=False).to(self.device)
        with torch.no_grad():
            if index == 0:
                mems = None
            else:
                cur = node
                mems = cur.mems
                while mems is None:
                    cur = cur.parent
                    mems = cur.mems
                    if mems is None:
                        token_ids = torch.cat((cur.state.token_ids, token_ids), dim=-1)


            outputs = self.model(token_ids, past_key_values=mems, return_dict=True)

        logits = outputs.logits
        logits = logits[:, -1, :len(self.tokenizer.vocab)]
        logits[:, self.thought_idx] = -100
        new_mems = outputs.past_key_values

        emitted_tokens = node.emitted_tokens if node is not None else {}

        #  index = torch.LongTensor(list(emitted_tokens.keys())).view(1, -1).to(logits.device)
        #  values = torch.tensor(list(emitted_tokens.values()), dtype=logits.dtype).view(1, -1).to(logits.device)
        indicator = torch.ones_like(logits) * self.args.temperature

        logits /= indicator
        logits = top_k_logits(logits, top_k=self.args.top_k, top_p=self.args.top_p)
        next_token_probs = F.softmax(logits, dim=-1)

        return next_token_probs, new_mems

def load_verifier(verifier_type, verifier_name):
    if verifier_type == "bert":
        verifier_model = AutoModelForMaskedLM.from_pretrained(verifier_name)
        verifier_tokenizer = BertTokenizer.from_pretrained(verifier_name)
    elif verifier_type == "gpt":
        verifier_model = AutoModelForCausalLM.from_pretrained(verifier_name)
        #  verifier_tokenizer = GPT2Tokenizer.from_pretrained(verifier_name, use_fast=True)
        verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_name)
    elif verifier_type == "deberta":
        verifier_model = DebertaV2ForMaskedLM.from_pretrained(verifier_name)
        verifier_tokenizer = DebertaV2Tokenizer.from_pretrained(verifier_name, use_fast=True)
    try:
        verifier_head = torch.load(os.path.join(verifier_name, "verifier_head.pth"))
    except:
        print("verifier_head.pth does not exist, random initialize one!")
        verifier_head = nn.Linear(1, 1, bias=True)

    if verifier_tokenizer.pad_token is None:
        verifier_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert "pad_token" in verifier_tokenizer.special_tokens_map
    if "[QUES]" not in verifier_tokenizer.vocab:
        print("Model is not trained on modified data!")
    verifier_tokenizer.add_tokens(['[QUES]', '[ANS]', '[THOUGHT]', '[VERIFIER]'])
    if "<|endoftext|>" not in verifier_tokenizer.vocab:
        verifier_tokenizer.add_tokens(["<|endoftext|>"])
    if verifier_model.config.vocab_size < len(verifier_tokenizer):
        verifier_model.resize_token_embeddings(new_num_tokens=len(verifier_tokenizer))

    return verifier_model.half().to(device), verifier_tokenizer, verifier_head.half().to(device)

def main(args):
    print(vars(args))

    data = DataProcessor._read_jsonl(args.data)
    for ex in data:
        ex['answer'] += "<|endoftext|>"
    #  {"question": "[QUES]Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n", "ground_truth": "[THOUGHT]Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n[ANS] 18", "solution": "[THOUGHT]Janet harvests 16 * 3 = <<16*3=48>>48 eggs every day for breakfast.\nShe harvests 48 * 4 = <<48*4=192>>192 muffins for her friends.\nShe sells 192 * 2 = <<192*2=384>>384 dollars for those muffins.\nThus, she makes 384 * $2 = $<<384*2=768>>768 at the farmers� market every day.\n[ANS] 768", "is_correct": false, "question_id": "0"}
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if model.config.vocab_size < len(tokenizer):
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    model = model.half().to(device)
    # --------------------------roll out verifier-------------------------------
    verifier_model, verifier_tokenizer, verifier_head = load_verifier(args.verifier_type, args.verifier_name)
    # --------------------------expand verifier-------------------------------
    expand_verifier_model, expand_verifier_tokenizer, expand_verifier_head = load_verifier(args.expand_verifier_type, args.expand_verifier_name)
    #  if model.config.vocab_size < len(tokenizer):
    #      model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    with jsonlines.open(args.data_name + "-" + args.timestamp + "-mcts_verifier_file.jsonl" + args.split, 'a', flush=True) as f:
        for idx, sample in enumerate(data):
            question = "[QUES]" + sample['question'] + "\n[THOUGHT]"
            answer = sample['answer'].replace("####", "[ANS]")
            ground_truth = extract_answer(answer)
            #  print("ground_truth", ground_truth)

            input_token_ids = tokenizer(question, return_tensors="pt").to(device).input_ids.view(1, -1)

            mcts = MCTS(model, tokenizer, args, device, verifier_model, verifier_head, verifier_tokenizer, expand_verifier_model, expand_verifier_head, expand_verifier_tokenizer, input_token_ids=input_token_ids, scalar=0.2)
            mcts.search()
            sample['question'] = "[QUES]" + sample['question'] + "\n"
            sample['ground_truth'] = "[THOUGHT]" + sample['answer'].replace("####", "[ANS]")
            del sample['answer']

            for case in mcts.good_cases:
                prediction = extract_answer(case[1])
                #  print("predicition", prediction)
                score, roll_out_verifier_score, ippl = case[0], case[2], case[3]
                if ground_truth == prediction:
                    print(f"Question {sample['question_id']}, correct prediction: ", case)
                f.write({**sample, "solution": case[1], "verifier_score": str(score), "is_correct": ground_truth == prediction, "roll_out_verifier_score": roll_out_verifier_score, "ippl": ippl})


if __name__ == "__main__":
    import argparse
    from data_preprocess import DataProcessor
    from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForMaskedLM, AutoTokenizer, BertTokenizer, DebertaV2Tokenizer, DebertaV2ForMaskedLM
    import torch
    import numpy as np
    import random
    parser = argparse.ArgumentParser()
    parser = add_gsm8k_args(parser)
    parser = add_common_ctrl_args(parser)
    parser = add_mcts_args(parser)
    args, argv = parser.parse_known_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    assert "pad_token" in tokenizer.special_tokens_map
    if "[QUES]" not in tokenizer.vocab:
        print("Model is not trained on modified data!")
    tokenizer.add_tokens(['[QUES]', '[ANS]', '[THOUGHT]', '[VERIFIER]'])
    print(f"Type of tokenizer: {type(tokenizer)}")
    #  EQUALS_TOKENS = set([28, 796, 47505])
    EQUALS_TOKENS = set(tokenizer.convert_tokens_to_ids(["=", "Ġ=", ")="]))
    main(args)

