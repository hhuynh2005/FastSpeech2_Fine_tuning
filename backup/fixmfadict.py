import os

org_filename = 'lexicon/vi-mfa.txt'
fixed_filename = 'lexicon/fvi-mfa.txt'

with open(org_filename, 'r', encoding='utf-8') as f:
    with open(fixed_filename, 'w', encoding='utf-8') as g:
        dic = {}
        for line in f:
            parts = line.split('\t')
            if parts[0].strip() not in dic:
                dic[parts[0].strip()] = ' '
                g.write(line.strip() + ' \n')

print("done!")
