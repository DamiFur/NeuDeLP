import pandas as pd
import re
import random
import config
from sklearn.model_selection import train_test_split

class LiteralConvertor:
    def __init__(self):
        self.stored_reps = {}
        self.last_num = 2
    def get_rep(self, literal):
        literal = literal.strip()
        neg_multiplier = 1
        if literal.startswith("~"):
            literal = literal.replace("~", "")
            neg_multiplier = -1
        if literal in self.stored_reps:
            return self.stored_reps[literal] * neg_multiplier
        else:
            self.last_num += 1
            self.stored_reps[literal] = self.last_num
            return self.last_num * neg_multiplier


convertor = LiteralConvertor()


def convert_to_tensor(row, max_arg_size=config.ARGUMENT_SIZE, include_blocking=False):
    arg1 = row['arg1']
    arg2 = row['arg2']
    defeater = row['defeater']

    # If DeLP-Gen script is used to generate arguments and programs, arg2 is always the defeater. We need to randomize this, otherwise the NN is going to learn to always choose the second argument
    arguments_list = [arg1, arg2]
    random.shuffle(arguments_list)
    arg1 = arguments_list[0]
    arg2 = arguments_list[1]

    # Get a numeric value indicating which of the two arguments is the defeater
    if include_blocking and defeater == 'blocking':
        output = 2
    elif arg2 == defeater:
        output = 1
    else:
        output = 0

    arg1_input = fix_length(process_argument(arg1, convertor), max_arg_size=max_arg_size)
    arg2_input = fix_length(process_argument(arg2, convertor), max_arg_size=max_arg_size)

    # print("Argument 1:")
    # print(arg1)
    # print(arg1_input)
    # print("Argument 2:")
    # print(arg2)
    # print(arg2_input)

    input = arg1_input + [config.ARGSEP] + arg2_input

    return input, output
    # print("-- INPUT --")
    # print(len(input))
    # print("-- OUTPUT --")
    # print(output)

def get_max_argument_size(csv):
    max_size = 0
    for row in csv.iterrows():
        if row[1]['arg1'] != "arg1":
            arg1 = row[1]['arg1']
            arg2 = row[1]['arg2']
            new_size = len(process_argument(arg1, convertor)) + len(process_argument(arg2, convertor)) + 1
            if new_size > max_size:
                max_size = new_size
    return max_size

def fix_length(encoded_argument, max_arg_size=config.ARGUMENT_SIZE):
    if len(encoded_argument) > max_arg_size:
        raise Exception("Argument too long. Length: {}".format(len(encoded_argument)))
    return encoded_argument + [config.FILLER] * (max_arg_size - len(encoded_argument))

def process_argument(argument, convertor):
    argument = argument.replace("[", "").replace("]", "")
    argument_parts = []
    argument_splitted = list(filter(None, sum(map(separate_facts, re.split(r'\(|\)', argument)),[])))
    for idx, rule in enumerate(argument_splitted):
        rule_splitted = re.split(r'-<|<-', rule)
        consequent = convertor.get_rep(rule_splitted[0])
        antecedent = []
        if len(rule_splitted) > 1 and not 'true' in rule_splitted:
            for lit in rule_splitted[1].split(","):
                antecedent.append(convertor.get_rep(lit))
        
        type_of_rule = config.FACT
        if 'true' in rule:
            type_of_rule = config.PRESUMPTION
        elif '-<' in rule:
            type_of_rule = config.DEFEASABLE
        elif '<-' in rule:
            type_of_rule = config.IMPLICATION

        argument_parts.append(consequent)
        argument_parts.append(type_of_rule)
        argument_parts += antecedent
        if idx != len(argument_splitted) -1:
            argument_parts.append(config.RULESEP) #TODO: Don't add when it's the last line
    return argument_parts

def separate_facts(argument_element):
    if '<' in argument_element:
        return [argument_element]
    else:
        return argument_element.split(',')

def get_train_test_datasets(complexity="simple", blocking=False, program_size=1000, output_size=2, presumptions="presumption_enabled", max_arg_size=-1):
    blocking_str = "blocking"
    if not blocking:
        blocking_str = "no_blocking"
    defs = pd.read_csv("datasets/{}-{}-{}-{}-{}.csv".format(complexity, blocking_str, program_size, output_size, presumptions))

    if max_arg_size == -1:
        max_arg_size = get_max_argument_size(defs)

    input_list = []
    for row in defs.iterrows():
        if row[1]['arg1'] != "arg1":
            input_list.append(convert_to_tensor(row[1], max_arg_size=max_arg_size, include_blocking=blocking))
        
        X = [input for input, output in input_list]
        Y = [output for input, output in input_list]
    
    ans = []
    for random_state in [42, 43, 44]:
        ans.append(train_test_split(X, Y, test_size = 0.2, random_state = random_state))

    return ans, max_arg_size