import torch

def get_random_problems(batch_size, node_cnt, problem_gen_params, mix_tsp=False, euc_data=False):

    scaler = problem_gen_params['scaler']

    if euc_data:
        ################################
        # Euclidean TSP (symmetric)
        ################################

        # Generate random 2D coordinates in [0, 1] range
        coords = torch.rand(size=(batch_size, node_cnt, 2))
        # shape: (batch, node_cnt, 2)

        # Compute pairwise Euclidean distances
        # problems[i,j] = ||coords[:,i] - coords[:,j]||_2
        diff = coords[:, :, None, :] - coords[:, None, :, :]   # (batch, node_cnt, node_cnt, 2)
        problems = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)  # (batch, node_cnt, node_cnt)

        # Diagonals are already ~0, but set explicitly to 0 for cleanliness
        problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

        return problems
        # shape: (batch, node, node)

    else:
        ################################
        # "tmat" type (ATSP)
        ################################

        int_min = problem_gen_params['int_min']
        int_max = problem_gen_params['int_max']

        problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt))

        # If requested, mix data: first half ATSP, second half symmetric TSP from (D + D^T)/2
        # if mix_tsp:
        #     # For odd batch_size we take floor(batch_size/2) for the front half
        #     half = batch_size // 2
        #     half = 0 # 全部
        #     back = problems[half:]
        #     back_sym = (back + back.transpose(1, 2)) // 2
        #     problems[half:] = back_sym

        # shape: (batch, node, node)
        problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

        for k in range(node_cnt):
            problems = torch.minimum(
                problems,
                problems[:, :, k:k+1] + problems[:, k:k+1, :]
            )

        # Scale
        scaled_problems = problems.float() / scaler

        return scaled_problems
        # shape: (batch, node, node)

def get_random_problems_orig(batch_size, node_cnt, problem_gen_params):

    ################################
    # "tmat" type
    ################################

    int_min = problem_gen_params['int_min']
    int_max = problem_gen_params['int_max']
    scaler = problem_gen_params['scaler']

    problems = torch.randint(low=int_min, high=int_max, size=(batch_size, node_cnt, node_cnt))
    # shape: (batch, node, node)
    problems[:, torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    while True:
        old_problems = problems.clone()

        problems, _ = (problems[:, :, None, :] + problems[:, None, :, :].transpose(2,3)).min(dim=3)
        # shape: (batch, node, node)

        if (problems == old_problems).all():
            break

    # Scale
    scaled_problems = problems.float() / scaler

    return scaled_problems
    # shape: (batch, node, node)


def load_single_problem_from_file(filename, node_cnt, scaler):

    ################################
    # "tmat" type
    ################################

    problem = torch.empty(size=(node_cnt, node_cnt), dtype=torch.long)
    # shape: (node, node)

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except Exception as err:
        print(str(err))

    line_cnt = 0
    for line in lines:
        linedata = line.split()

        if linedata[0].startswith(('TYPE', 'DIMENSION', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_WEIGHT_SECTION', 'EOF')):
            continue

        integer_map = map(int, linedata)
        integer_list = list(integer_map)

        problem[line_cnt] = torch.tensor(integer_list, dtype=torch.long)
        line_cnt += 1

    # Diagonals to 0
    problem[torch.arange(node_cnt), torch.arange(node_cnt)] = 0

    # Scale
    scaled_problem = problem.float() / scaler

    return scaled_problem
    # shape: (node, node)