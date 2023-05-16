def syntax_occl_ECP(x):
    # keep only occluded
    x = [x for x in x if "occluded" in x]
    if x and "occluded" in x[0]:
        return int(x[0].replace("occluded>", "")) / 100
    elif x:
        print(x)
    else:
        return 0

def syntax_truncated_ECP(x):
    # keep only truncated
    x = [x for x in x if "truncated" in x]
    if x and "truncated" in x[0]:
        return int(x[0].replace("truncated>", "")) / 100
    elif x:
        print(x)
    else:
        return 0


def syntax_criteria_ECP(x, criteria):
    # "sitting-lying"
    # keep only truncated
    x = [x for x in x if criteria in x]
    if x and criteria in x[0]:
        return 1
    elif x:
        print(x, criteria)
    else:
        return 0
