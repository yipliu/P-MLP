import copy
from tqdm import tqdm

from lota.lota_utils.utils import loadjson


def load_data(datas, data_dir):
    """
    datas = {task_type: data}
    """
    newdatas = []
    paths = []

    for t, ds in datas.items():
        for d in ds:
            data_path = data_dir/d/"traj_data.json" 
            if data_path.is_file():
                paths.append(data_path)
                # with open(data_path) as f:
                #     data = json.load(f)
                data = loadjson(data_path)
                task_dir = data_path.parts[-3]
                data['task_dir']=task_dir
                newdatas.append(data)

            else:
                raise ValueError
    return newdatas


def convert_low_level_action_to_nl_skill(action, args, cur_obj, before_low_action):
    """
    If two consecutive low-level actions are both interactions, no `find`
    action is added because no navigation occurs between them.

    before_low_action 上一时刻的 action
    """

    from lota.src.alfred.utils import ithor_name_to_natural_word, find_indefinite_article

    nl_steps = []
    dl_steps = []
    ret_obj = None
    interaction = ['OpenObject', 'CloseObject', 'PutObject', 'PickupObject', 'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject']

    # flag
    is_before_interaction = before_low_action in interaction

    def obj_id_to_nl(s):
        return ithor_name_to_natural_word(s.split('|')[0])

    if action == 'OpenObject':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o and not is_before_interaction:
            nl_steps.append(f'find {find_indefinite_article(o)} {o}')
            dl_steps.append(f"find_{o}")
        ret_obj = o
        nl_steps.append(f'open the {o}')
        dl_steps.append(f'open_{o}')

    elif action == 'CloseObject':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o and not is_before_interaction:
            nl_steps.append(f'find {find_indefinite_article(o)} {o}')
            dl_steps.append(f"find_{o}")
        ret_obj = o
        nl_steps.append(f'close the {o}')
        dl_steps.append(f'close_{o}')
        
    elif action == 'PutObject':
        o_recep = obj_id_to_nl(args['receptacleObjectId'])
        # if cur_obj != o_recep and not is_before_interaction:
        if cur_obj != o_recep:
            nl_steps.append(f'find {find_indefinite_article(o_recep)} {o_recep}')
            dl_steps.append(f'find_{o_recep}')
        ret_obj = o_recep
        o = obj_id_to_nl(args['objectId'])
        nl_steps.append(f'put down the {o}') # ['find a counter top', 'put down the pot']
        dl_steps.append(f'putdown_{o}')


    elif action == 'PickupObject':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o and not is_before_interaction:
            nl_steps.append(f'find {find_indefinite_article(o)} {o}')
            dl_steps.append(f'find_{o}')
        ret_obj = o
        nl_steps.append(f'pick up the {o}')
        dl_steps.append(f'pickup_{o}')


    elif action == 'ToggleObjectOn':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o and not is_before_interaction:
            nl_steps.append(f'find {find_indefinite_article(o)} {o}')
            dl_steps.append(f'find_{o}')
        ret_obj = o
        nl_steps.append(f'turn on the {o}')
        dl_steps.append(f'turnon_{o}')


    elif action == 'ToggleObjectOff':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o and not is_before_interaction:
            nl_steps.append(f'find {find_indefinite_article(o)} {o}')
            dl_steps.append(f'find_{o}')
        ret_obj = o
        nl_steps.append(f'turn off the {o}')
        dl_steps.append(f'turnoff_{o}')


    elif action == 'SliceObject':
        o = obj_id_to_nl(args['objectId'])
        if cur_obj != o and not is_before_interaction:
            nl_steps.append(f'find {find_indefinite_article(o)} {o}')
            dl_steps.append(f'find_{o}')
        ret_obj = o
        nl_steps.append(f'slice the {o}')
        dl_steps.append(f'slice_{o}')

    else:
        pass

    return nl_steps, dl_steps, ret_obj


def annotationData(datas, newSplit):
    """ Obtain traj_data
    newSplit: after random.sample      
    Just for (action, object).
    Images are collected separately by `GT_Planner`. 
    """

    newdatas = []

    for e in tqdm(datas):
        # exclude pick_two_obj_and_place type
        if e['task_type'] == 'pick_two_obj_and_place':
            continue

        NL_steps, DL_steps = [], [] 
        skip_this_sample = False
   

        cur_obj = None
        before_low_action = '' # low_action at t-1

        for s_i, s in enumerate(e['plan']['low_actions']):
            action = s['api_action']['action']
            args = s['api_action']

            if s_i!= 0:
                before_low_action = e['plan']['low_actions'][s_i-1]['api_action']['action']
            else:
                before_low_action = ''

            try:
                nl_skill, dl_skill, cur_obj = convert_low_level_action_to_nl_skill(action, args, cur_obj, before_low_action)
            except ValueError:
                skip_this_sample = True
                break


            if nl_skill:
                NL_steps.extend(nl_skill) 
                DL_steps.extend(dl_skill) # action_obj


        if skip_this_sample or len(NL_steps) <= 0:
            continue

        else:
            # 增加 stop_action 和 stop_img
            NL_steps.append("stop")
            DL_steps.append("stop_none")


        new_traj_data = {
            'pddl_params': copy.deepcopy(e['pddl_params']),
            'plan': copy.deepcopy(e['plan']),
            'scene': copy.deepcopy(e['scene']),
            # 'task': copy.deepcopy(e['task']),   # rewards 
            'task_id': copy.deepcopy(e['task_id']), # 'trial_T20190907_174127_043461'
            'task_type': copy.deepcopy(e['task_type']),        # 'look_at_obj_in_light'
            'task_dir': copy.deepcopy(e['task_dir']), # 'look_at_obj_in_light-AlarmClock-None-DeskLamp-301'
            'turk_annotations': {'anns': copy.deepcopy(e['turk_annotations']['anns'])},
            'split': newSplit,
            'NL_steps': NL_steps,
            'DL_steps': DL_steps, # 包含 find action
        }

        newdatas.append(new_traj_data)

    return newdatas


def export_alfred_examples(newsplitdatasets, data_dir):
    """
    datasets = {'train': xx, 'valid':, ...}
    """
    newDatasets = {}

    # split_new: split_old
    data_split = {"train": "train",
                  "valid": "train",
                  "test_seen": "valid_seen",
                  "test_unseen": "valid_unseen"}

    
    for split, datas in newsplitdatasets.items():
        print(f"=====Processing {split} split=====")
        # Step1: Load trajectories
        datas = load_data(datas, data_dir/data_split[split])

        print(f"{split}: {len(datas)}")
        # Step2: Convert each trajectory into high-level annotations.
        traj_datas = annotationData(datas, split)

        newDatasets[split]=traj_datas
    
    return newDatasets