import optuna
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import yaml
from optuna.samplers import TPESampler

# 설정한 하이퍼파라미터 랜덤 값 반환하기
def get_params(config, trial=None):
    if trial is None:
        return config  # trial이 없으면 config를 그대로 반환

    hp_config = config['hyperparameter_optimization']
    for param, settings in hp_config['parameters'].items():
        if settings['type'] == 'log_uniform':
            value = trial.suggest_loguniform(param, float(settings['min']), float(settings['max'])) # loguniform 값 반환
        elif settings['type'] == 'uniform':
            value = trial.suggest_uniform(param, settings['min'], settings['max']) # uniform 값 반환
        elif settings['type'] == 'int':
            value = trial.suggest_int(param, settings['min'], settings['max']) # int 값 반환
        elif settings['type'] == 'categorical':
            value = trial.suggest_categorical(param, settings['values']) # categorical 값 반환
        else:
            continue  # 알 수 없는 타입은 건너뜁니다

        # 새로운 값으로 config 업데이트
        if param in config['training']:
            config['training'][param] = value
        else:
            config[param] = value

    return config