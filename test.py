"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import argparse

import numpy as np
import torch

from solver import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.cuda.set_device(1)

import os
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from scipy.stats import hmean
from utils import cuda, grid2gif
from model_share import Generator_fc, Generator_fc_dsprites
from dataset1 import load_data
from PIL import Image
import torch.nn as nn
import functools
import networks
from torchvision import transforms

import dataset1 as dset

from common import Evaluator

torch.cuda.set_device(1)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Solver(object):
    def __init__(self, args):
        #GPU
        self.use_cuda = args.cuda and torch.cuda.is_available()

        #dataset and Z-space
        if args.dataset.lower() == 'ilab_20m' or args.dataset.lower() == 'ilab_20m_custom':
            self.nc = 3
            self.z_dim = 100 #'dimension of the latent representation z'
            # id: 0~60; back: 60~80; pose: 80~100
            self.z_pose_dim = 20  # 'dimension of the pose latent representation in z'
            self.z_back_dim = 20  # 'dimension of the background latent representation in z'
            self.z_id_dim = self.z_dim - self.z_pose_dim - self.z_back_dim # 'dimension of the id latent representation in z'

        elif args.dataset.lower() == 'utzappos':
            self.nc = 3
            self.z_dim =300
            #'dimension of the latent representation z'
            # att: 0~64; obj: 64-128;
            self.z_att_dim =150# 'dimension of the z_content (letter) latent representation in z'

            self.z_obj_dim = 150# 'dimension of the z_size latent representation in z'
            self.z_att_start_dim = 0
            self.z_obj_start_dim =150


        self.dataset = args.dataset
        if args.train: # train mode
            self.train = True
        else: # test mode
            self.train = False
            args.batch_size = 1
            self.pretrain_model_path = args.pretrain_model_path
            self.test_img_path = args.test_img_path
        self.batch_size = args.batch_size
        #self.data_loader = return_data(args) ### key


        self.trainset = dset.CompositionDataset(
            root = os.path.join(args.DATA_FOLDER, args.data_dir),
            phase = 'train',
            split = args.splitname,
            pair_dropout = args.pair_dropout,
            update_features = args.update_features,
            train_only = args.train_only,
            model = args.image_extractor,
            num_negs = args.num_negs,
            open_world = args.open_world,
        )
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers)

        testset = dset.CompositionDataset(
            root=os.path.join(args.DATA_FOLDER, args.data_dir),
            phase='test',
            split=args.splitname,
            model=args.image_extractor,
            subset=args.subset,
            update_features=args.update_features,
            open_world=args.open_world
        )
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size= args.batch_size,
            shuffle=False,
            num_workers=args.workers)



        # model training param
        self.topk = args.topk
        self.bias = args.bias
        self.g_conv_dim = args.g_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.norm_layer = get_norm_layer(norm_type=args.norm)
        self.max_iter = args.max_iter
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.Evaluator=Evaluator
        self.graph_init = args.graph_init
        self.gr_emb = args.gr_emb
        self.lambda_combine = args.lambda_combine
        self.lambda_unsup = args.lambda_unsup
        if args.dataset.lower() == 'dsprites':
            self.Autoencoder = Generator_fc_dsprites(self.nc, self.g_conv_dim, self.g_repeat_num, self.z_dim)
        else:
            self.Autoencoder = Generator_fc(self.nc, self.g_conv_dim, self.g_repeat_num, self.z_dim, self.graph_init,
                                            self.gr_emb)
        self.Autoencoder = self.Autoencoder.cuda()
        self.auto_optim = optim.Adam(self.Autoencoder.parameters(), lr=self.lr,
                                     betas=(self.beta1, self.beta2))

        # log and save
        self.log_dir = './checkpoints/' + args.viz_name
        self.model_save_dir = args.model_save_dir
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_combine_sup = None
        self.win_combine_unsup = None

        self.gather_step = args.gather_step
        self.gather = DataGather()
        self.display_step = args.display_step


        self.resume_iters = args.resume_iters
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.save_step = args.save_step
        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.Evaluator=Evaluator(testset)

    def restore_model(self, resume_iters):
        """Restore the trained generator"""
        if resume_iters == 'pretrained':
            print('Loading the pretrained models from  {}...'.format(self.pretrain_model_path))
            self.Autoencoder.load_state_dict(torch.load(self.pretrain_model_path, map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{} '".format(self.pretrain_model_path))
        else: # not test
            print('Loading the trained models from step {}...'.format(resume_iters))
            Auto_path = os.path.join(self.model_save_dir, self.viz_name, '{}-Auto.ckpt'.format(resume_iters))
            self.Autoencoder.load_state_dict(torch.load(Auto_path, map_location=lambda storage, loc: storage))
            print("=> loaded checkpoint '{} (iter {})'".format(self.viz_name, resume_iters))

    # For Fonts dataset

    def test_fonts(self):
        # self.net_mode(train=True)
        # load pretrained model
        self.restore_model('pretrained')

        torch.cuda.empty_cache()
        img_embed = torch.empty(0, 300).cuda()
        accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []

        self.Autoencoder.eval()

        path = self.graph_init
        graph = torch.load(path)
        self.embeddings = graph['embeddings'].cuda()
        adj = graph['adj']

        for index, sup_package in enumerate(self.testloader):
            A_img = sup_package[0]
            A_img = Variable(cuda(A_img, self.use_cuda))
            ## 1. A B C seperate(first400: id last600 background)

            with torch.no_grad():
                A_recon, A_z = self.Autoencoder(A_img)
            # A_z_m = A_z[:, 0:128].add(A_z[:, 128:256])
            img_embed = torch.cat((img_embed, A_z))
            attr_truth, obj_truth, pair_truth = sup_package[4], sup_package[5], sup_package[6]
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

        # union_list = list(set(self.testloader.dataset.train_pairs).union(set(self.testloader.dataset.test_pairs)))
        #
        # with torch.no_grad():
        current_embeddings = self.Autoencoder.gcn(self.embeddings)
        allpair_embedding = current_embeddings[28:144, :]
        allpair_embedding = allpair_embedding.cuda()
        img_embed = img_embed.cuda()
        allpair_embedding = allpair_embedding.permute(1, 0)
        score = torch.matmul(img_embed, allpair_embedding)
        scores = {}

        for itr, pair in enumerate(self.testloader.dataset.pairs):
            scores[pair] = score[:, self.testloader.dataset.all_pair2idx[pair]]

        all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to(
            'cpu'), torch.cat(all_pair_gt).to('cpu')

        # Calculate best unseen accuracy
        results = self.Evaluator.score_manifold_model(scores, all_obj_gt, bias=self.bias, topk=self.topk)
        stats = self.Evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, scores,
                                                    topk=self.topk)

        result = ''
        for key in stats:
            result = result + key + '  ' + str(round(stats[key], 4)) + '| '

        result = result

        print('Results')
        print(result)
        return results
class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    combine_sup_loss=[],
                    combine_unsup_loss=[],
                    images=[],
                    combine_supimages=[],
                    combine_unsupimages=[],
                    test=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()
class Evaluator:

    def __init__(self, dset):

        self.dset = dset

        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=5):  # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''

        def get_pred_from_scores(_scores, topk):
            '''
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            '''
            _, pair_pred = _scores.topk(topk, dim=1)  # sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                                  self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(scores.shape[0], 1)  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        # Unbiased setting

        # Open world setting --no mask, all pairs of the dataset
        results.update({'open': get_pred_from_scores(scores, topk)})
        results.update({'unbiased_open': get_pred_from_scores(orig_scores, topk)})
        # Closed world setting - set the score for all Non test pairs to -1e10,
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({'closed': get_pred_from_scores(closed_scores, topk)})
        results.update({'unbiased_closed': get_pred_from_scores(closed_orig_scores, topk)})

        # Object_oracle setting - set the score to -1e10 for all pairs where the true object does Not participate, can also use the closed score
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        oracle_obj_scores[~mask] = -1e10
        oracle_obj_scores_unbiased = orig_scores.clone()
        oracle_obj_scores_unbiased[~mask] = -1e10
        results.update({'object_oracle': get_pred_from_scores(oracle_obj_scores, 1)})
        results.update({'object_oracle_unbiased': get_pred_from_scores(oracle_obj_scores_unbiased, 1)})

        return results

    def score_clf_model(self, scores, obj_truth, topk=5):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to('cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])  # Return only attributes that are in our pairs
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)  # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.cuda()

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=5):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''

        results = {}
        mask = self.seen_mask.repeat(scores.shape[0], 1)  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        _, pair_pred = closed_scores.topk(topk, dim=1)  # sort returns indices of k largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                              self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk=1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = attr_truth.to('cpu'), obj_truth.to('cpu'), pair_truth.to('cpu')

        pairs = list(
            zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(unseen_ind)

        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk])
            obj_match = (obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk])

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            ### Calculating class average accuracy

            # local_score_dict = copy.deepcopy(self.test_pair_dict)
            # for pair_gt, pair_pred in zip(pairs, match):
            #     # print(pair_gt)
            #     local_score_dict[pair_gt][2] += 1.0 #increase counter
            #     if int(pair_pred) == 1:
            #         local_score_dict[pair_gt][1] += 1.0

            # # Now we have hits and totals for classes in evaluation set
            # seen_score, unseen_score = [], []
            # for key, (idx, hits, total) in local_score_dict.items():
            #     score = hits/total
            #     if bool(self.seen_mask[idx]) == True:
            #         seen_score.append(score)
            #     else:
            #         unseen_score.append(score)

            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)

            return attr_match, obj_match, match, seen_match, unseen_match, \
                   torch.Tensor(seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = ['_attr_match', '_obj_match', '_match', '_seen_match', '_unseen_match', '_ca', '_seen_ca',
                    '_unseen_ca']
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        ##################### Match in places where corrent object
        obj_oracle_match = (
                    attr_truth == predictions['object_oracle'][0][:, 0]).float()  # object is already conditioned
        obj_oracle_match_unbiased = (attr_truth == predictions['object_oracle_unbiased'][0][:, 0]).float()

        stats = dict(obj_oracle_match=obj_oracle_match, obj_oracle_match_unbiased=obj_oracle_match_unbiased)

        #################### Closed world
        closed_scores = _process(predictions['closed'])
        unbiased_closed = _process(predictions['unbiased_closed'])
        _add_to_dict(closed_scores, 'closed', stats)
        _add_to_dict(unbiased_closed, 'closed_ub', stats)

        #################### Calculating AUC
        scores = predictions['scores']
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][unseen_ind]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats['closed_unseen_match'].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats['closed_seen_match'].mean())
        unseen_match_max = float(stats['closed_unseen_match'].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to('cpu') for k, v in allpred.items()}
        obj_truth = obj_truth.to('cpu')

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(scores, obj_truth, bias=bias, topk=topk)
            results = results['closed']  # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats['biasterm'] = float(bias_term)
        stats['best_unseen'] = np.max(unseen_accuracy)
        stats['best_seen'] = np.max(seen_accuracy)
        stats['AUC'] = area
        stats['hm_unseen'] = unseen_accuracy[idx]
        stats['hm_seen'] = seen_accuracy[idx]
        stats['best_hm'] = max_hm
        return stats
def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    solver = Solver(args)

    if args.train:
        if args.dataset.lower() == 'utzappos':
            solver.train_fonts()
    else:
        if args.dataset.lower() == 'utzappos':
            solver.test_fonts()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GZS-Net')
    '''
    Need modify base on your dataset
    '''

    parser.add_argument('--train', default=False, type=str2bool, help='train: True or test: False')
    parser.add_argument('--dataset', default='utzappos', help='mitstates|zappos')
    parser.add_argument('--data_dir', default='ut-zap50k1', help='local path to data root dir from ')
    parser.add_argument('--resume_iters', type=int, default=0,
                        help='resume training from this step, e,g, 10000 or test model selection')
    parser.add_argument('--viz_name', default='utzappos', type=str, help='visdom env name')
    parser.add_argument('--image_size', type=int, default=224, help='crop size for the ilab dataset')
    parser.add_argument('--pretrain_model_path', default='./checkpoints/pretrained_models/bestAUC.ckpt',
                        type=str,
                        help='pretrain model path')
    parser.add_argument('--workers', type=int, default=0, help="Number of workers")

    parser.add_argument('--update_features', action='store_true', default=True,
                        help='If specified, train feature extractor')
    parser.add_argument('--splitname', default='compositional-split-natural', help="dataset split")
    parser.add_argument('--train_only', action='store_true', default=True, help='Optimize only for train pairs')
    parser.add_argument('--pair_dropout', type=float, default=0.0, help='Each epoch drop this fraction of train pairs')
    parser.add_argument('--DATA_FOLDER', default='F:/mcl/202219/', help='local path to data root dir from')
    parser.add_argument('--image_extractor', default='resnet18', help='Feature extractor model')
    parser.add_argument('--num_negs', type=int, default=1,
                        help='Number of negatives to sample per positive (triplet loss)')
    parser.add_argument('--open_world', action='store_true', default=False, help='perform open world experiment')

    parser.add_argument('--test_img_path', default='./checkpoints/test_imgs/', type=str,
                        help='pretrain model path')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1000000, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    # model params
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--g_repeat_num', type=int, default=1,
                        help='number of residual blocks in G for encoder and decoder')
    parser.add_argument('--lambda_combine', type=float, default=1, help='weight for lambda_combine')
    parser.add_argument('--lambda_unsup', default=0, type=float, help='lambda_recon')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--lr', default=5.0e-05, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    parser.add_argument('--num_workers', default=32, type=int, help='dataloader num_workers')
    # log
    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    # save model
    # parser.add_argument('--model_save_dir', default='checkpoints', type=str, help='output directory')
    parser.add_argument('--model_save_dir', default='./checkpoints/pretrained_models/', type=str,
                        help='output directory')
    parser.add_argument('--subset', action='store_true', default=False,
                        help='test on a 1000 image subset (debug purpose)')
    parser.add_argument('--gather_step', default=5000, type=int,
                        help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=5000, type=int,
                        help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=100, type=int,
                        help='number of iterations after which a checkpoint is saved')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--graph_init', default='utzappos-graph.t7',
                        help='filename, file from which initializing the nodes and adjacency matrix of the graph')

    parser.add_argument("--gr_emb", default='d4096,d', help="graph layers config")
    parser.add_argument('--test_batch_size', type=int, default=32, help="Batch size at test/eval time")
    parser.add_argument('--bias', type=float, default=1e3, help='Bias value for unseen concepts')
    parser.add_argument('--topk', type=int, default=1, help="Compute topk accuracy")

    args = parser.parse_args()
    main(args)