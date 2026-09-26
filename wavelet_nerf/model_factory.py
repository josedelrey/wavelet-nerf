"""One model constructor shared by training and evaluation."""
from wavelet_nerf.configuration import validate_resolved_config
from wavelet_nerf.models import LegacyNeRF, NeRF, Siren, WaveletNeRF


def model_hyperparameters(config):
    model_type = config['model_type']
    if model_type == 'nerf':
        keys = ('pos_encoding_dim', 'dir_encoding_dim', 'hidden_dim')
        result = {key: config[key] for key in keys}
        if config['baseline_version'] == 'reference':
            result['num_importance'] = config['num_importance']
        return result
    if model_type == 'siren':
        return {**{key: config[key] for key in ('num_layers', 'sigma_mul', 'rgb_mul', 'w0', 'hidden_w0')},
                'hidden_dim': config['siren_hidden_dim'],
                'dir_encoding_dim': config['siren_dir_encoding_dim']}
    return {**{key: config[key] for key in ('input_scale', 'weight_scale', 'alpha', 'beta', 'omega0', 'normalized')},
            'in_features': config['wave_in_features'], 'hidden_dim': config['wave_hidden_dim'],
            'num_layers': config['wave_num_layers'], 'dir_encoding_dim': config['wave_dir_encoding_dim']}


def create_model(config):
    """Construct an ordinary model from validated, resolved experiment settings."""
    config = validate_resolved_config(config)
    if config['model_type'] == 'nerf':
        constructor = NeRF if config['baseline_version'] == 'reference' else LegacyNeRF
    else:
        constructor = {'siren': Siren, 'wavelet': WaveletNeRF}[config['model_type']]
    return constructor(**model_hyperparameters(config))
