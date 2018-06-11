order = ["right", "left", "down", "up", "start", "select", "B", "A"]
base_offset = 3

def parse_fm2(file):
    for line in file.readlines():
        if line[0] != "|":
            continue
        yield {order[i].upper():line[base_offset+i]!="." for i in range(8)}