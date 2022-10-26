import pandas as pd
import re
import random

# Types of implication: defeasable implication, regular implication, fact or presumption
FACT = -1
PRESUMPTION = 1
DEFEASABLE = 1
IMPLICATION = -1

# Separation within and between arguments
RULESEP = 2
ARGSEP = -2

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


def convert_to_tensor(row):
    arg1 = row['arg1']
    arg2 = row['arg2']
    defeater = row['defeater']

    # If DeLP-Gen script is used to generate arguments and programs, arg2 is always the defeater. We need to randomize this, otherwise the NN is going to learn to always choose the second argument
    arguments_list = [arg1, arg2]
    random.shuffle(arguments_list)
    arg1 = arguments_list[0]
    arg2 = arguments_list[1]

    # Get a numeric value indicating which of the two arguments is the defeater
    output = 1 if arg2 == defeater else 0

    arg1_input = process_argument(arg1, convertor)
    arg2_input = process_argument(arg2, convertor)
    
    print("Argument 1:")
    print(arg1)
    print(arg1_input)
    print("Argument 2:")
    print(arg2)
    print(arg2_input)

    # TODO: Arguments should be of the same length
    input = arg1_input + [ARGSEP] + arg2_input
    print("-- INPUT --")
    print(input)
    print("-- OUTPUT --")
    print(output)


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
        
        type_of_rule = FACT
        if 'true' in rule:
            type_of_rule = PRESUMPTION
        elif '-<' in rule:
            type_of_rule = DEFEASABLE
        elif '<-' in rule:
            type_of_rule = IMPLICATION

        argument_parts.append(consequent)
        argument_parts.append(type_of_rule)
        argument_parts += antecedent
        if idx != len(argument_splitted) -1:
            argument_parts.append(RULESEP) #TODO: Don't add when it's the last line
    return argument_parts

def separate_facts(argument_element):
    if '<' in argument_element:
        return [argument_element]
    else:
        return argument_element.split(',')

defs = pd.read_csv("defs.csv")


defs.apply(convert_to_tensor, axis=1)