# adopt from domainbed repository

import argparse
import collections
import re
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

import datasets
import hparams_registry
import algorithms
from libs import misc
from libs.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default="../../../data")
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--serial', type=str, default="D")

    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--cudanumber', type=int, default=0)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output/")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true', default=True)
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None
    
    output_dir = args.output_dir + str(args.cudanumber)
    jour = time.strftime("%Y-%m-%d-jour-%H-%M-%S", time.localtime())+'-'+str(args.serial)
    output_dir += jour
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda:{}".format(args.cudanumber)
    else:
        device = "cpu"

    # 
    # hparams include batch size, lr, weight decay
    # dataset is a list of pytorch dataset object
    #print(args.test_envs)
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
        
    else:
        raise NotImplementedError


    in_splits = []
    out_splits = []
    uda_splits = []
    # dataset is a list of pytorch dataset
    for env_i, env in enumerate(dataset):
        uda = []
        # for each env(a pytorch dataset), randomly divide it into 2 parts
        if 'c2group' in hparams.keys():
            if env_i%2 == 0:
                in_ = dataset[env_i]
                out = dataset[env_i+1]
            else:
                continue
        else:
            out, in_ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
        
        # weights are the coefficients in the emperical loss fonction
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None

        # Elements in in_splits and out_splits are one-to-one corresponded
        # they are datasets representing environments
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # train loader contains only in_splits
    # only train model on train loaders
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS,
        index=i)
        for i, (env, env_weights) in enumerate(in_splits)]
        #if i not in args.test_envs]
    print(len(train_loaders))
    #print(hparams['batch_size'])
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]
    
    # eval loader contains the entire dataset, including both train and validation loader
    # when iterate over eval_loaders, the first several datasets are training loaders, 
    # following datasets are evaluation loaders
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]

    # names are not big issues 
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    #print(eval_loader_names)
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)

    # if the first input is not None, result is the first, otherwise the second
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = 1 #args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    hparams["n_steps"] = n_steps

    # algorithm encapsulates models
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    
    # looks useless
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    
    # zip takes in iterables and outputs a tuple
    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)

    # checkpoint_vals are used for saving statistics on the training process
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    
    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(output_dir, filename))


    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        # this generates the dataset list
        minibatches_device = train_loaders#[(x.to(device), y.to(device))
            #for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        
        # the most important line, rest are saving issues
        # step vals is {'loss': loss item, 'penalty': penalty item}
        # step_vals = algorithm.update(minibatches_device, uda_device, device=device)
        step_vals = algorithm.update(minibatches_device, device=device, id_up=step)


        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        
        # if checkpoint_freq, print result
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            
            # evaluation accuracy
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            totacci = 0.
            totacco = 0.
            toti = 0.
            toto = 0.
            for name, loader, weights in evals:
                id_in_name = re.findall(r"\d+\.?\d*",name)
                if 'regression' in hparams.keys():
                    acc = misc.meanseloss(algorithm, loader, weights, device, int(id_in_name[0]))
                    results[name+'_loss'] = acc
                else:
                    acc = misc.accuracy(algorithm, loader, weights, device, int(id_in_name[0]))
                    results[name+'_acc'] = acc
                if "in" in name:
                    totacci += acc
                    toti += 1
                else:
                    totacco += acc
                    toto += 1
            results['average_acc_in'] = totacci/toti
            results['average_acc_out'] = totacco/toto

            # names are sorted before being printed
            # so keys are env0_in_acc, env0_out_acc, ..., rather than env0_in_acc, env1_in_acc, ...
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                # print 12 items in one row of results_keys
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })
            
            # update the json file
            epochs_path = os.path.join(output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1

            # renew the checkpoint_vals dictionary
            checkpoint_vals = collections.defaultdict(lambda: [])

            # Not saving the model
            #if args.save_model_every_checkpoint:
            #    save_checkpoint(f'model_step{step}.pkl')

    #save_checkpoint('model.pkl')

    with open(os.path.join(output_dir, 'done'), 'w') as f:
        f.write('done')
    