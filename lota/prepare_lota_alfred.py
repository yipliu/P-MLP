import os
import sys
sys.path.append('/path/to/your/root_dir')
import copy
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from lota.lota_utils.builder import make_TrainDev, load_tasks,alfredStatistics
from lota.lota_utils.annotation import export_alfred_examples
from lota.lota_utils.utils import dumpjson


def get_lota_alfred(aflred_dir, dev_traj_nums=300, out_dir=None):

    # 1. re-splitting ALFRED 
    Train, train_statistic, Dev, dev_statistic = make_TrainDev(aflred_dir, dev_traj_nums)
    TestSeen, test_seen_statistic   = load_tasks(aflred_dir, 'valid_seen')   
    TestUnseen, test_unseen_statistic = load_tasks(aflred_dir, 'valid_unseen')

    # 2. computing dataset statistics
    NewSplitDatasets = {'train': Train, 'valid': Dev, 'test_seen': TestSeen, 'test_unseen': TestUnseen}
    alfredStatistics(out_dir, train_statistic, dev_statistic, test_seen_statistic, test_unseen_statistic)


    # 3. annotations
    trajDatasets = export_alfred_examples(NewSplitDatasets, aflred_dir)
    return trajDatasets

def tasktypeStatistics(datasets):
    tasktypeTrajs = defaultdict(lambda: [])
    for trajdata in datasets:
        #data = loadjson(trajdata)
        datatype = trajdata['task_type']
        tasktypeTrajs[datatype].append(trajdata)

    statistics = {}
    for tasktype, datas in tasktypeTrajs.items():
        statistics[tasktype] = len(datas)
    return statistics


def datasetStatistics(trajDatasets, out_dir):
    """数据统计分析

    - 每个split 的 traj 数量
    - split 中每类任务的  traj 数量
    """

    statistics = {}
    statistics['total'] = {}
    for split, datasets in trajDatasets.items():
        # train/valid/test 的 traj 数量
        statistics['total'][split] = len(datasets)
    
        # 获取每个 task type 的数量
        statistics[split] = tasktypeStatistics(datasets)


    out_file = out_dir / 'lyp_dataset_trajs_info.json'
    if out_file.is_file():
        os.remove(str(out_file))
    dumpjson(statistics, out_file)


if __name__ == '__main__':
    aflred_dir = Path('LLMTaskPlanning_datas/alfred/data/json_2.1.0')
    out_dir = Path('/LLMTaskPlanning_datas/data')
    dev_traj = 300
    text_dir = out_dir / 'text_datasets' 

    # 1. LoTa-ALFRED creation
    trajDatasets=get_lota_alfred(aflred_dir, dev_traj, out_dir)


    # 2. Save LoTa-ALFRED trajectory data.
    for split, datas in trajDatasets.items():
        text_data_dir = text_dir /split  # text_datasets/train/valid/test_seen/test_unseen
        text_data_dir.mkdir(parents=True, exist_ok=True)

        print(f"---Saving: {split} split-----")
        for data in tqdm(datas):
            traj_dir = text_data_dir/data['task_dir'] / data['task_id'] 
            traj_dir.mkdir(parents=True, exist_ok=True) 
            traj_name = traj_dir / f"traj_data.json" 
            dumpjson(data, traj_name)        

    # 3. Save dataset statistics.
    datasetStatistics(trajDatasets, out_dir)

    # 4. save split file and  gt_models
    # Use task_info as the index since language instructions are not unique.
    print("---- Building GT models ----")
    # split_file = defaultdict(lambda: [])
    gt_models = {}
    for k, dataset in trajDatasets.items():
        gt_models[k] = {}
        for d in tqdm(dataset):
            task_info = f"{d['task_dir']}/{d['task_id']}"
            gt_models[k][task_info] = copy.deepcopy(d['DL_steps'])

        print(f"{k}: {len(gt_models[k])}")    

