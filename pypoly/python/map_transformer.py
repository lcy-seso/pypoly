# this function generates the code to pack data for 'map'
# note: the order in dims matters
def domain(dims):
    # dim1 dim2 ... dimn
    # i1 i2 ... in
    # c1 c2 ... cn
    # c1 = i1 + i2 + ... + in
    # ci = i_{k-1}, k >= 2
    gen_code = []
    n = len(dims)
    idx_str = 'idx'
    max_val_str = 'max_val'
    gen_code.append('max_val = ' + ' + '.join(dims) + ' - ' + str(n))
    gen_code.append('\n')
    indent = '    '
    # dep_data refers to data related to the dependence flow
    dep_data_strs = []
    # about the order of each dim: except the last dim, other dims are ordered lexicographically
    for i, dim in enumerate(dims):
        cur_bound_str = 'bound' + str(i)
        dep_data_str = 'dep_data_' + str(i)
        dep_data_strs.append(dep_data_str)
        gen_code.append(cur_bound_str + ' = ' + max_val_str + ' - ' + dim + ' + 1')
        # since the dim of the hyperplan is n-1, the last dim share the same structure with the (n-1)th dim in a reverse order
        # as a result, parameters of 'concat' and 'gather' are in a dual way.
        # 3 cases
        gen_code.append('if ' + dim + ' < ' + cur_bound_str + ':')
        # 3 cases
        gen_code.append(indent + 'if ' + idx_str + ' < ' + dim + ':')
        if i < n - 1:
            gen_code.append(indent + indent + dep_data_str + ' = concat(init_state, prev_state, dim={})'.format(i))
        else:
            gen_code.append(indent + indent + dep_data_str + ' = concat(prev_state, init_state, dim={})'.format(i - 1))
        gen_code.append(indent + 'elif ' + idx_str + ' < ' + cur_bound_str + ' + 1:')
        if i < n - 1:
            gen_code.append(indent + indent + dep_data_str + ' = concat(init_state, gather(prev_state, end={}, dim={}), dim={})'.format(dim + ' - 1', i, i))
        else:
            gen_code.append(indent + indent + dep_data_str + ' = concat(gather(prev_state, start=1, dim={}), init_state, dim={})'.format(i - 1, i - 1))
        gen_code.append(indent + 'else:')
        if i < n - 1:
            gen_code.append(indent + indent + dep_data_str + ' = gather(prev_state, end={}, dim={})'.format(dim + ' - 1', i))
        else:
            gen_code.append(indent + indent + dep_data_str + ' = gather(prev_state, start=1, dim={})'.format(i - 1))
        gen_code.append('elif ' + dim + ' = ' + cur_bound_str + ':')
        # 2 cases
        gen_code.append(indent + 'if ' + idx_str + ' < ' + dim + ':')
        if i < n - 1:
            gen_code.append(indent + indent + dep_data_str + ' = concat(init_state, prev_state, dim={})'.format(i))
        else:
            gen_code.append(indent + indent + dep_data_str + ' = concat(prev_state, init_state, dim={})'.format(i - 1))
        gen_code.append(indent + 'else:')
        if i < n - 1:
            gen_code.append(indent + indent + dep_data_str + ' = gather(prev_state, end={}, dim={})'.format(dim + ' - 1', i))
        else:
            gen_code.append(indent + indent + dep_data_str + ' = gather(prev_state, start=1, dim={})'.format(i - 1))
        gen_code.append('else:')
        # 3 cases
        gen_code.append(indent + 'if ' + idx_str + ' < ' + cur_bound_str + ':')
        if i < n - 1:
            gen_code.append(indent + indent + dep_data_str + ' = concat(init_state, prev_state, dim={})'.format(i))
        else:
            gen_code.append(indent + indent + dep_data_str + ' = concat(prev_state, init_state, dim={})'.format(i - 1))
        gen_code.append(indent + 'elif ' + idx_str + ' < ' + dim + ' + 1:')
        gen_code.append(indent + indent + dep_data_str + ' = prev_state')
        gen_code.append(indent + 'else:')
        if i < n - 1:
            gen_code.append(indent + indent + dep_data_str + ' = gather(prev_state, start=1, dim={})'.format(i))
        else:
            gen_code.append(indent + indent + dep_data_str + ' = gather(prev_state, end={}, dim={})'.format(dims[i - 1] + ' - 1', i - 1))
        gen_code.append('\n')
    return gen_code

def test_stacked_lstm():
    dims = ['len(xs)', 'depth']
    print('\n'.join(domain(dims)).replace('\n\n', '\n'))

if __name__ == "__main__":
    test_stacked_lstm()