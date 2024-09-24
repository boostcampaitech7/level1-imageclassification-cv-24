import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

import os
import pandas as pd
from src.models.model_utils import get_model
from src.utils.data_loaders import get_test_loaders

def run(config):
    device = torch.device(config['device'])
    
    if config['ensemble']['use_ensemble']:
        models = []
        ensemble_path = os.path.join(config['paths']['output_dir'], 'ensemble_models.pth')
        ensemble_state_dicts = torch.load(ensemble_path)
        for i, state_dict in enumerate(ensemble_state_dicts):
            model = get_model(config['ensemble']['models'][i]).to(device)
            model.load_state_dict(state_dict)
            models.append(model)
    else:
        model = get_model(config).to(device)
        model.load_state_dict(torch.load(os.path.join(config['paths']['output_dir'], "best_model1.pth")))
        models = [model]
        
    model = get_model(config).to(device)
    
    test_loader = get_test_loaders(config)

    model.load_state_dict(
        torch.load(
            os.path.join(config['paths']['save_dir'],"best_model1.pth"),
            map_location='cpu'
        )
    )

    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)

            logits = model(images)
            logits = F.softmax(logits,dim=1)
            preds = logits.argmax(dim=1)

            predictions.extend(preds.cpu().detach().numpy())
    
    # 모든 클래스에 대한 예측 결과를 하나의 문자열로 합침
    test_info = pd.read_csv(config['data']['test_info_file'])

    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info
    test_info.to_csv(os.path.join(config['paths']['output_dir'],"output.csv"), index=False)
    
