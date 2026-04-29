from pathlib import Path
import random
from collections import defaultdict
import os

from lota.lota_utils.utils import dumpjson


def countFilesInFold(fold, datasetName, statistic=False):
    """Count the number of trajectories in a dataset split.
    """
    traj_count = 0
    statistic_value = None
    if statistic:
        statistic_value = {}

    for key in fold:
        traj_count += len(fold[key])

    print(f"{datasetName}: {traj_count} trajs")
    if statistic_value is not None:
        statistic_value['total'] = traj_count

    tasks = defaultdict(lambda: [])
    task_count = 0

    for k, v in fold.items():
        task_name = k.split("-")[0]
        tasks[task_name].extend(v) 

    for k, v in tasks.items():
        task_count += len(v)
        print(f"{k}: {len(v)}")
        
        if statistic_value is not None:
            statistic_value[k] = len(v)
        
    assert  traj_count == task_count   

    return tasks, task_count, statistic_value




def make_TrainDev(data_dir, dev_traj):
    """Split the original ALFRED training set into new training and validation sets.
    """
    outTrain = {}
    outDev = {}

    # Step 1: Load all traj_data.json
    print("Loading all train traj_data.json files...")
    train_dir = Path(data_dir) / 'train'
    train_files = list(train_dir.rglob("*traj_data.json")) 

    filenamesByProblem = {}
    for filenameWithPath in train_files:
        filenameParts = filenameWithPath.parts
        filename = f"{filenameParts[-3]}/{filenameParts[-2]}" 
        filenamePrefix = filenameParts[-3]
        fileTask = filenamePrefix.split("-")[0]

        # Exclude the Pick Two & Place task category.
        if fileTask == 'pick_two_obj_and_place':
            continue

        if (filenamePrefix not in filenamesByProblem):
            filenamesByProblem[filenamePrefix] = list()
        filenamesByProblem[filenamePrefix].append(filename) # 


    # Step 2: Separate into train/dev sets
    for key in filenamesByProblem:
        outTrain[key] = list()
        outDev[key] = list()

        samples = filenamesByProblem[key]
        if (len(samples) == 0):
            # Do nothing
            pass

        elif (len(samples) == 1):
            # Keep single-trajectory instances in the training set.
            outTrain[key].append( samples[0] )

        else:
            # Randomly select one trajectory for validation.
            Dev_indicies = random.sample(range(len(samples)), 1)
            for idx in range(len(samples)):
                if idx in Dev_indicies:
                    outDev[key].append( samples[idx] )
                else:
                    outTrain[key].append( samples[idx] )

    # Count trajectories in the initial train/validation split.
    Train, countTrain, _ = countFilesInFold(outTrain, 'train', False)
    Dev, countDev, _ = countFilesInFold(outDev, 'val', False)


    # Step3: 重新采样

    new_Dev = {}
    if countDev > dev_traj:
        # 对 Dev 再次进行 缩减

        dev = {'look_at_obj_in_light': 29, 'pick_and_place_simple': 46,
               'pick_and_place_with_movable_recep': 34,'pick_clean_then_place_in_recep': 37,
               'pick_cool_then_place_in_recep': 38,'pick_heat_then_place_in_recep': 34}
        
        total_dev = sum(list(dev.values()))
        
        for k, v in Dev.items():
            new_Dev[k] = list()

            new_sample_indicies = random.sample(range(len(v)), int((dev[k]/total_dev)*dev_traj) )

            for idx in range(len(v)):
                if idx in new_sample_indicies:
                    new_Dev[k].append(v[idx])
                else:
                    Train[k].append(v[idx])

    # Summary statistics
    NewTrain, countTrain, train_statistic_value = countFilesInFold(Train, 'train', True)
    NewDev, countDev, dev_statistic_value = countFilesInFold(new_Dev, 'val', True)

    return NewTrain, train_statistic_value, NewDev, dev_statistic_value



def load_tasks(data_dir, split):
    task_paths = defaultdict(list)
    base_path = data_dir / split
    
    paths = sorted(base_path.glob("**/traj_data.json"))
    
    right_paths = [p for p in paths if 'pick_two_obj_and_place' not in str(p)]
    print(f"{split} traj (without pick_two_obj_and_place): {len(right_paths)}")
    statistic_value = {}
    statistic_value['total']=len(right_paths)

    for p in paths:
        pParts = p.parts
        pName = f"{pParts[-3]}/{pParts[-2]}"
        pTask = pParts[-3].split("-")[0]

        if pTask == 'pick_two_obj_and_place':
            continue
        task_paths[pTask].append(pName)

    for k, v in task_paths.items():
        statistic_value[k]=len(v)

    return task_paths, statistic_value  



def alfredStatistics(out_dir, train_statistic, dev_statistic, test_seen_statistic, test_unseen_statistic):
    alfred_statistic = {}
    alfred_statistic['total'] = {}
    alfred_statistic['train'] = {}
    alfred_statistic['valid_seen'] = {}
    alfred_statistic['valid_unseen'] = {}
   
    alfred_statistic['total']['valid_seen'] = test_seen_statistic['total'] 
    alfred_statistic['total']['valid_unseen'] = test_unseen_statistic['total'] 

    
    for k, v in train_statistic.items():
        if k == 'total':
            alfred_statistic[k]['train'] = train_statistic[k] + dev_statistic[k]
        else:
            alfred_statistic['train'][k] = train_statistic[k] + dev_statistic[k]

    for k, v in test_seen_statistic.items():
        if k == 'total':
            alfred_statistic[k]['valid_seen'] = test_seen_statistic[k]
        else:
            alfred_statistic['valid_seen'][k] = test_seen_statistic[k]

    for k, v in test_unseen_statistic.items():
        if k == 'total':
            alfred_statistic[k]['valid_unseen'] = test_unseen_statistic[k]
        else:
            alfred_statistic['valid_unseen'][k] = test_unseen_statistic[k]


    out_file = out_dir / 'alfred_dataset_trajs_info.json'
    if out_file.is_file():
        os.remove(str(out_file))
    dumpjson(alfred_statistic, out_file)
