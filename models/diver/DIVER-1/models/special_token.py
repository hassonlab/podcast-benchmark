import torch
import torch.nn as nn

class TokenManager(nn.Module):
    def __init__(self, d_model, input_format="BCND"):  
        super(TokenManager, self).__init__()
        self.d_model = d_model
        self.special_tokens = nn.ParameterDict() 
        self.input_format = input_format
        self.seq_dim = 2
        self.RAN_PREPEND = False

        token_prepend_list = ['N_token', 'C_token', 'NC_token']
        for token_name in token_prepend_list: 
            self.create_token(token_name)

    def create_token(self, name, token_shape=None): 
        token_shape = (1, 1, self.d_model)
        token = torch.randn(*token_shape) * 0.02
        self.special_tokens[name] = nn.ParameterDict({
            "param": nn.Parameter(token, requires_grad=True),
            "insert_pos": None,  
        })    
    
    def _expand_token(self, token, x_shape, expand_along_dimension=None):        
        if self.input_format == "BCND":
            B, C, N, D = x_shape
            if expand_along_dimension is None:
                return token.unsqueeze(0).expand(B, -1, -1, -1)
            elif expand_along_dimension == "C":
                assert token.shape[0] == 1, "Token must have c_count=1 to expand along C"
                return token.expand(B, C, -1, -1)
            elif expand_along_dimension == "N":
                assert token.shape[1] == 1, "Token must have n_count=1 to expand along N"
                return token.expand(B, -1, N, -1)
            else:
                raise ValueError(f"expand_along_dimension must be None, 'N', or 'C'")
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")
        
    def prepend_x_with_tokens(self, x, mode :str, n_token_name :str = None, c_token_name :str = None, nc_token_name : str = None, 
                              expand_n_along_dimensions : bool = False, expand_c_along_dimensions :bool = False) : 
        assert self.RAN_PREPEND == False, "prepend_x_with_tokens should be called only once per forward pass"
        self.RAN_PREPEND = True
        
        x_org_shape = x.shape
        n_token = self.special_tokens[n_token_name]["param"] 
        c_token = self.special_tokens[c_token_name]["param"]
        nc_token = self.special_tokens[nc_token_name]["param"]

        if n_token is not None : 
            expand_along_dimension = "C" if expand_n_along_dimensions else None
            n_token = self._expand_token(n_token, x.shape, expand_along_dimension=expand_along_dimension)
        if c_token is not None : 
            expand_along_dimension = "N" if expand_c_along_dimensions else None
            c_token = self._expand_token(c_token, x.shape, expand_along_dimension=expand_along_dimension)
        if nc_token is not None:
            nc_token = self._expand_token(nc_token, x.shape, expand_along_dimension=None) 
            
        x = torch.cat([n_token, x], dim=2)
        nc_and_c_token = torch.cat([nc_token, c_token], dim=2) 
        x = torch.cat([nc_and_c_token, x], dim=1)

        self.PREPEND_PARAMS = {'mode' : mode, 'org_x_shape' : x_org_shape, \
            'n_token_shape_during_prepend' : n_token.shape if n_token is not None else None, \
            'c_token_shape_during_prepend' : c_token.shape if c_token is not None else None, \
            'nc_token_shape_during_prepend' : nc_token.shape if nc_token is not None else None, \
        }
        return x 

    def extract_prepended_x(self, x) :
        assert self.RAN_PREPEND == True, "extract_prepended_x should be called after prepend_x_with_tokens"
        self.RAN_PREPEND = False 

        n_token_shape = self.PREPEND_PARAMS['n_token_shape_during_prepend']
        n_token_count = n_token_shape[2] if n_token_shape is not None else 0 
        c_token_shape = self.PREPEND_PARAMS['c_token_shape_during_prepend']
        c_token_count = c_token_shape[1] if c_token_shape is not None else 0 
        nc_token_shape = self.PREPEND_PARAMS['nc_token_shape_during_prepend']
                
        n_token_output = x[:, c_token_count:, :n_token_count].clone() if n_token_count > 0 else None
        c_token_output = x[:, :c_token_count, n_token_count:].clone() if c_token_count > 0 else None
        nc_token_output = x[:, :c_token_count, :n_token_count].clone() if nc_token_shape is not None else None
                
        x = x[:, c_token_count:, n_token_count:]
                
        return {
            "n_position_features": n_token_output,
            "c_position_features": c_token_output,
            "nc_position_features": nc_token_output,
            "org_x_position_features": x
        }