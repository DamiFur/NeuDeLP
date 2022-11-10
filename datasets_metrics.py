import pandas as pd

def output_size(blocking):
    if blocking == 'blocking':
        return 3
    else:
        return 2

for complexity in ['simple', 'complex']:
    for blocking in ['blocking', 'no_blocking']:
        total_pairs = 0
        total_programs = 0
        for program_size in [200, 500, 1000]:
            for presumptions in ['presumption_enabled', 'presumption_disabled']:
                filename = "datasets/{}-{}-{}-{}-{}.csv".format(complexity, blocking, program_size, output_size(blocking), presumptions)
                file_content = pd.read_csv(filename)
                file_pairs = len(file_content) - program_size + 1
                total_pairs += file_pairs
                total_programs += program_size
                print("{} total arguments pairs: {}".format(filename, file_pairs))
        print("{} - {}:".format(complexity, blocking))
        print("  total pairs: {}".format(total_pairs))
        print("  total programs: {}".format(total_programs))
        print("  AVERAGE: {}".format(total_pairs/total_programs))
