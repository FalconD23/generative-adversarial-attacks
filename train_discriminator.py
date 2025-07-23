import os
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from clearml import Task

from src.config import get_criterion, get_disc_list, get_model
from src.data import MyDataset, load_data, transform_data
from src.estimation.estimators import AttackEstimator
from src.training.train import DiscTrainer
from src.utils import fix_seed, save_config, save_compiled_config,weights_from_clearml_by_name

warnings.filterwarnings("ignore")

CONFIG_NAME = "train_disc_config"
CONFIG_PATH = "config"

torch.cuda.empty_cache()
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):

    if cfg["test_run"]:
        print("ATTENTION!!!! Results will not be saved. Set param test_run=False")
        logger = None
    else:
        model_start_name = 'model_{}_{}_{}_attack_{}'.format(
            cfg["attack_model"]["name"],
            cfg["model_id_attack"],
            cfg["dataset"]["name"],
            cfg["attack"]["short_name"],
        )
        if cfg['log_clearml'] and cfg['author'] == '':
            raise ValueError("You need to set your name in config")

        model_add_name = ''
        for param in cfg['attack']['named_params']:
            model_add_name += '__{}={}'.format(
                param,
                round(cfg['attack']['attack_params'][param], 4)
            )

        save_path = os.path.join(cfg["save_path"], model_start_name + model_add_name)
        save_config(save_path, CONFIG_PATH, CONFIG_NAME, CONFIG_NAME)
        save_compiled_config(cfg, save_path, model_start_name + model_add_name)

    fix_seed(cfg['model_id'])

    augmentator = (
        [instantiate(trans) for trans in cfg["transform_data"]]
        if cfg["transform_data"] else None
    )

    X_train, y_train, X_test, y_test = load_data(cfg["dataset"]['name'])

    if len(set(y_test)) > 2:
        return None

    X_train, X_test, y_train, y_test = transform_data(
        X_train,
        X_test,
        y_train,
        y_test,
        slice_data=cfg["slice"],
    )

    train_loader = DataLoader(
        MyDataset(X_train, y_train),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    test_loader = DataLoader(
        MyDataset(X_test, y_test),
        batch_size=cfg["batch_size"],
        shuffle=False,
    )

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    if cfg['load_weights_classifier']:
        project_name = cfg['project_weights']
        task_name = f"model_{cfg['model']['name']}_{cfg['model_id_attack']}_{cfg['dataset']['name']}"

        path = weights_from_clearml_by_name(project_name=project_name, task_name=task_name)
        attack_model_path = os.path.join(path)
    else:

        attack_model_path = os.path.join(
            cfg["model_folder"],
            f"model_{cfg['model']['name']}_{cfg['model_id_attack']}_{cfg['dataset']['name']}.pt",
        )

    attack_model = get_model(
        cfg["attack_model"]["name"],
        cfg["attack_model"]["params"],
        path=attack_model_path,
        device=device,
        train_mode=cfg["attack_model"]["attack_train_mode"],
    )

    criterion = get_criterion(cfg["criterion_name"], cfg["criterion_params"])
    if cfg['load_weights_disc']:
        path = cfg['project_weights_disc']
        
    else:
        path = cfg["disc_path"]
    if cfg["use_disc_check"]:
        disc_check_list = get_disc_list(
            model_name=cfg["disc_model_check"]["name"],
            model_params=cfg["disc_model_check"]["params"],
            list_disc_params=cfg["list_check_model_params"],
            device=device,
            path=path,
            train_mode=False,
            from_clearml=cfg['load_weights_disc']
        )
    else:
        disc_check_list = None

    estimator = AttackEstimator(
        disc_check_list,
        cfg["metric_effect"],
        cfg["metric_hid"],
        batch_size=cfg["estimator_batch_size"],
    )

    attack_params = dict(cfg["attack"]["attack_params"])
    attack_params["model"] = attack_model
    attack_params["criterion"] = criterion
    attack_params["estimator"] = estimator

    if "list_reg_model_params" in cfg["attack"]:
        attack_params["disc_models"] = get_disc_list(
            model_name=cfg["disc_model_reg"]["name"],
            model_params=cfg["disc_model_reg"]["params"],
            list_disc_params=cfg["attack"]["list_reg_model_params"],
            device=device,
            path=path,
            train_mode=cfg["disc_model_reg"]["attack_train_mode"],
            from_clearml=cfg['load_weights_disc']
        )

    const_params = {
            "attack_params": attack_params,
            "print_every": cfg["print_every"],
            "device": device,
            "seed": cfg['model_id'],
            "train_self_supervised": cfg["train_self_supervised"],
    }

    if cfg["enable_optimization"]:
        disc_trainer = DiscTrainer.initialize_with_optimization(
            train_loader, test_loader, cfg["optuna_optimizer"], const_params
        )

        if not cfg["test_run"]:
            model_add_name = ''
            for param in cfg['attack']['named_params']:
                model_add_name += '__{}={}'.format(
                    param,
                    round(getattr(disc_trainer.attack, param), 4)
                )
    else: 
        const_params["attack_name"] = cfg["attack"]["name"]
        trainer_params = dict(cfg["training_params"])
        trainer_params.update(const_params)
        disc_trainer = DiscTrainer.initialize_with_params(**trainer_params)


    if not cfg["test_run"]:
        model_save_name = model_start_name + model_add_name
        exp_name = cfg['exp_name'][1:] if cfg['exp_name'][0] == '_' else cfg['exp_name']
        if cfg['log_clearml']:
            task = Task.init(
                project_name=cfg['clearml_project'],
                task_name=model_save_name,
                tags=[
                    cfg["attack_model"]["name"],
                    cfg["dataset"]["name"],
                    cfg["attack"]["short_name"],
                    exp_name,
                    cfg['author'],
                ]
            )
           
            task.upload_artifact(artifact_object=f'{cfg["save_path"]}/{model_save_name}/{model_save_name}.pt', name='model_weights.pt')
        else:
            task = None
        logger = SummaryWriter(cfg["save_path"] + "/tensorboard")

    disc_trainer.train_model(train_loader, test_loader, augmentator, logger)
    # if cfg['load_weights_classifier']:
    #     os.remove(attack_model_path)
    # else:
    #     pass


    if not cfg["test_run"]:
        print("Saving")
        new_save_path = os.path.join(cfg["save_path"], model_save_name)
        disc_trainer.save_result(new_save_path, model_save_name, task)
        if cfg['load_weights_classifier']:
            os.remove(attack_model_path)
        if cfg['delete_weights_disc']:
            target_folder = 'loaded_clearml/disc_weights/'

            for file_name in os.listdir(target_folder):
                file_path = os.path.join(target_folder, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

if __name__ == "__main__":
    main()
    # print("OK-"*20)
