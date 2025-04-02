def load_parameters():
    
    """Define the bounds and plausible bounds for parameters."""
    
    default_values = {
        'alphaOmega': 0.1,
        'alphaSS': 0.1,
        'alphaSAS': 0.1,
        'betaPred': 1.0,
        'biasArbitrator': 0.0,
        'slopeOmega': 0.0,
    }
        
    bounds_list = {
        'alphaOmega': [0.001,0.999],
        'alphaSS': [0.001,0.999],
        'alphaSAS': [0.001,0.999],
        'betaPred': [-100.0, 100.0],
        'epsilon': [0.001, 0.999],
        'biasArbitrator': [-10.0, 10.0],
        'slopeOmega': [-50.0, 50.0],
    }

    plausible_bounds_list = {
        'alphaOmega': [0.05, 0.95],
        'alphaSS': [0.05, 0.95],
        'alphaSAS':[0.05, 0.95],
        'epsilon': [0.01, 0.25],
        'betaPred': [-1.0, 25.0],
        'biasArbitrator': [-5.0, 5.0],
        'slopeOmega': [-25.0, 25.0],
    }

    prior_shapes={
        'alphaOmega': 'Trapezoidal',
        'alphaSS': 'Trapezoidal',
        'alphaSAS': 'Trapezoidal',
        'epsilon': 'Trapezoidal',
        'betaPred': 'SmoothBox',
        'biasArbitrator': 'SmoothBox',
        'slopeOmega': 'SmoothBox',
    }
    return bounds_list, plausible_bounds_list, default_values, prior_shapes

