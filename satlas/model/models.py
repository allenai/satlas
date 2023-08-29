import satlas.model.model
import satlas.model.classify

def get_model(info):
    model_type = info['config'].get('Name', 'multihead4')
    if model_type == 'classify':
        return satlas.model.classify.Model(info)
    else:
        return satlas.model.model.Model(info)