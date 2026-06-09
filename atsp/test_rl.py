import torch
from RLmodel import ATSPModel

def test_model():
    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 2,
        'qkv_dim': 16,
        'sqrt_qkv_dim': 16**(1/2),
        'head_num': 8,
        'init': 'svd',
        'att_type': 'normal',
        'logit_clipping': 10,
        'ff_hidden_dim': 256,
        'ms_hidden_dim': 16,
        'ms_layer1_init': (3/3)**(1/2),
        'ms_layer2_init': (3/16)**(1/2),
        'eval_type': 'softmax',
        'one_hot_seed_cnt': 50,
        'max_steps': 20,
    }
    
    model = ATSPModel(**model_params)
    print("Model works!")
    

if __name__ == "__main__":
    test_model()