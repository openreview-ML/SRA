# external libs
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import random
from os.path import join as ospj
from glob import glob
# torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
# local libs
from utils import get_norm_values, chunks
from itertools import product
import scipy.sparse as sp


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        img = Image.open(ospj(self.root_dir, img)).convert('RGB')  # We don't want alpha
        return img


def dataset_transform(phase, norm_family='imagenet'):
    '''
        Inputs
            phase: String controlling which set of transforms to use
            norm_family: String controlling which normaliztion values to use

        Returns
            transform: A list of pytorch transforms
    '''
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform


def filter_data(all_data, pairs_gt, topk=5):
    '''
    Helper function to clean data
    '''
    valid_files = []
    with open('/home/ubuntu/workspace/top' + str(topk) + '.txt') as f:
        for line in f:
            valid_files.append(line.strip())

    data, pairs, attr, obj = [], [], [], []
    for current in all_data:
        if current[0] in valid_files:
            data.append(current)
            pairs.append((current[1], current[2]))
            attr.append(current[1])
            obj.append(current[2])

    counter = 0
    for current in pairs_gt:
        if current in pairs:
            counter += 1
    print('Matches ', counter, ' out of ', len(pairs_gt))
    print('Samples ', len(data), ' out of ', len(all_data))
    return data, sorted(list(set(pairs))), sorted(list(set(attr))), sorted(list(set(obj)))


# Dataset class now
def load_data(path="D:/Gt/CZSL/czsl_gt/gt/data/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""

    idx_features_labels = np.genfromtxt("{}{}.embedding".format(path, dataset),
                                        dtype=np.dtype(float))
    features = sp.csr_matrix(idx_features_labels[:, 1:301], dtype=np.float32)

    # build symmetric adjacency matrix

    features = torch.FloatTensor(np.array(features.todense()))

    return features

class CompositionDataset(Dataset):
    '''
    Inputs
        root: String of base dir of dataset
        phase: String train, val, test
        split: String dataset split
        subset: Boolean if true uses a subset of train at each epoch
        num_negs: Int, numbers of negative pairs per batch
        pair_dropout: Percentage of pairs to leave in current epoch
    '''

    def __init__(
            self,
            root,
            phase,
            split='compositional-split',
            model='resnet18',
            norm_family='imagenet',
            subset=False,
            num_negs=1,
            pair_dropout=0.0,
            update_features=True,
            return_images=False,
            train_only=False,
            open_world=False,
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.num_negs = num_negs
        self.pair_dropout = pair_dropout
        self.norm_family = norm_family
        self.return_images = return_images
        self.update_features = update_features
        self.feat_dim = 512 if 'resnet18' in model else 2048  # todo, unify this  with models
        self.open_world = open_world

        self.attrs, self.objs, self.pairs, self.train_pairs, \
        self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()
        self.full_pairs = list(product(self.attrs, self.objs))

        # Clean only was here
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        if self.open_world:
            self.pairs = self.full_pairs

        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        if train_only and self.phase == 'train':
            print('Using only train pairs')
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}
        else:
            print('Using all pairs')
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            print('Using all data')
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('Invalid training phase')

        self.all_data = self.train_data + self.val_data + self.test_data
        print('Dataset loaded')
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data)))

        if subset:
            ind = np.arange(len(self.data))
            ind = ind[::len(ind) // 1000]
            self.data = [self.data[i] for i in ind]

        # Keeping a list of all pairs that occur with each object
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj,_) in self.train_data + self.test_data if obj == _obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj,_) in self.train_data if obj == _obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Load based on what to output
        self.transform = dataset_transform(self.phase, self.norm_family)
        self.loader = ImageLoader(ospj(self.root, 'images'))
        if not self.update_features:
            feat_file = ospj(root, model + '_featurers.t7')
            print(f'Using {model} and feature file {feat_file}')
            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file, model)
            self.phase = phase
            activation_data = torch.load(feat_file)
            self.activations = dict(
                zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)
            print('{} activations loaded'.format(len(self.activations)))

    def parse_split(self):
        '''
        Helper function to read splits of object atrribute pair
        Returns
            all_attrs: List of all attributes
            all_objs: List of all objects
            all_pairs: List of all combination of attrs and objs
            tr_pairs: List of train pairs of attrs and objs
            vl_pairs: List of validation pairs of attrs and objs
            ts_pairs: List of test pairs of attrs and objs
        '''

        def parse_pairs(pair_list):
            '''
            Helper function to parse each phase to object attrribute vectors
            Inputs
                pair_list: path to textfile
            '''
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [line.split() for line in pairs]
                pairs = list(map(tuple, pairs))

            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.root, self.split, 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.root, self.split, 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.root, self.split, 'test_pairs.txt')
        )

        # now we compose all objs, attrs and pairs
        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        '''
        Helper method to read image, attrs, objs samples

        Returns
            train_data, val_data, test_data: List of tuple of image, attrs, obj
        '''
        data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))

        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['_image'], instance['attr'], instance['obj'], instance['set']
            # if image=="engraved_redwood/SAUNZEE-Custom-Wood-Sandblasted-Sign-Sandblast-Sign-Jagged-edge-Signs-3D-Engraved-Carved-Wood-Sign-Routed-Sign-Sand-blasted-Sign-Visual-Merchandising-Signage-Retail-Signage-Paradise-Found-Honolulu-Hawaii.jpg":
            #     image=image.replace(image, 'engraved_redwood/SAUNZEE.jpg');
            # if image=="large_garage/Exterior-Paint-Schemes-ouse-design-architecture-american-exterior-paint-schemes-completion-with-cream-brown-paint-and-large-garage-door-also-concrete-drive-way-decorative-exterior-paint-schemes-completion.jpg":
            #     image=image.replace(image, 'large_garage/Exterior.jpg');
            # if image=="tiny_bathroom/interior-with-yellow-shower-curtain-other-design-contemporary-tiny-bathroom-with-small-place-with-yellow-shower-curtain-also-chroom-heads-shower-design-tiny-bathroom-designs-inspirations-images-amazing-shower-curtain-des.jpg":
            #     image=image.replace(image, 'tiny_bathroom/interior.jpg');
            # if image=="large_bathroom/bathroom-architecture-freestanding-acrylic-bathtub-and-round-freestanding-ceramic-vessel-sink-also-recessed-ceiling-lighting-in-large-bathroom-layouts-design-fascinating-large-bathroom-layouts-design.jpg":
            #     image=image.replace(image, 'large_bathroom/freestanding.jpg');
            # if image=="cut_tile/decoration-tile-designs-and-patterns-bath-floor-tile-limestone-floor-tile-interior-design-help-tiles-design-for-hall-somany-floor-tiles-tiles-for-walls-tiles-pattern-floor-tiles-designs-cut-tile-tile-edg-1058x793.jpg":
            #     image=image.replace(image, 'cut_tile/decoration.jpg');
            # if image=="large_bathroom/bathroom-attractive-large-bathroom-layouts-with-modular-bath-vanities-with-marble-on-top-and-large-wide-mirror-combined-with-luxury-bathtub-ideas-and-exclusive-tile-floor-inspiration-breathtaking-larg.jpg":
            #     image=image.replace(image, 'large_bathroom/marble.jpg');
            # if image=="large_bathroom/bathroom-luxury-large-bathroom-layouts-with-decorative-bathtub-with-column-style-also-excellent-wooden-cabinet-and-antique-mirror-above-tower-rack-ideas-breathtaking-large-bathroom-layouts-ideas.jpg":
            #     image=image.replace(image, 'large_bathroom/luxury.jpg');
            # if image=="wrinkled_dog/?url=http%3A%2F%2Fs3.amazonaws.com%2Fassets.prod.vetstreet.com%2F32%2F34%2Ff0b501be417781c55897ba886dc4%2Fbloodhound-AP-7YWIPF-590sm4913.jpg":
            #     image = image.replace(image, 'wrinkled_dog/amazonaws.jpg');
            # if image=="unpainted_ceiling/Charming-Home-Style-Applied-Perfect-Combine-Brick-Wall-Flower-Above-It-Black-Tiles-Spacious-Hallway-To-Another-Room-Wooden-Door-Unpainted-Ceiling-Cozy-White-Chair-With-Footrest-Rounded-Side-Table-Ornament.jpg":
            #     image = image.replace(image, 'unpainted_ceiling/Charming.jpg');
            # if image=="tiny_bathroom/bathroom-white-sink-plus-brown-wooden-storage-vanity-and-rectangular-soaking-bathtub-reversible-drain-combine-wall-mounted-polished-chrome-double-towel-bar-for-tiny-bathroom-ideas-delightful-tiny-bat.jpg":
            #     image = image.replace(image, 'tiny_bathroom/tiny.jpg');
            # if image=="ruffled_velvet/?url=http%3A%2F%2Fcdn.hgtvgardens.com%2F26%2F74%2Ff5ca4ce64b18a149091931449666%2F4019_078.jpg":
            #     image = image.replace(image, 'ruffled_velvet/hgtvgardens.jpg');
            # if image=="small_house/1_small_house_|_credit_-_Danny_Yahini.jpg":
            #     image = image.replace(image, 'small_house/credit.jpg');
            # if image=="crinkled_ribbon/proxy?container=onepick&gadget=a&rewriteMime=image%2F*&url=http%3A%2F%2Fi1118.photobucket.com%2Falbums%2Fk601%2FAdonijah73%2FCrafts%2Fcrinkleribbon027.jpg":
            #     image = image.replace(image, 'crinkled_ribbon/onepick.jpg');
            # if image=="crushed_sand/2005-06-15_12:54.36__02133.jpg":
            #     image=image.replace(image, 'crushed_sand/11.jpg');
            # if image=="burnt_truck/Pakistani%2Bpolice%2Bofficers%2Bexamine%2Ba%2Bburnt%2Btruck%2Btorched%2Bby%2Bsuspected%2Bmilitants%2Bin%2Ban%2Battack%2Bearly%2BWednesday,%2BJune%2B9,%2B2010,%2Bin%2BSangjani%2Bnear%2BIslamabad,%2BPakistan.jpg":
            #     image = image.replace(image, 'burnt_truck/Pakistani.jpg');
            # if image=="unpainted_wall/Cheerful-Modern-Industrial-Style-Staircase-Design-Decorating-Luxury-Home-Black-Tiles-Flooring-Small-Pendant-Lamp-Glass-Panels-Black-Framed-Shade-Trees-In-Garden-Unpainted-Wall-Combine-Marvelous-Parquet.jpg":
            #     image = image.replace(image, 'unpainted_wall/Cheerful.jpg');
            # if image=="thin_tile/interior-design-natural-stone-rock-wall-dimplex-fireplaces-living-room-set-up-siding-faux-fireplace-hearths-brick-ideas-stone-modern-thin-tile-direct-manufacturers-standing-antique-mantel-warehouse-contemporary-desig-fir.jpg":
            #     image = image.replace(image, 'thin_tile/interior-design.jpg');
            # if image=="thin_tile/decoration-thin-tile-ideas-refacing-a-panels-free-standing-slate-wall-kits-stone-faux-brick-realstone-veneerstone-diy-stone-brick-wall-cost-rock-panels-fake-faux-river-wall-decor-panels-wall-supplier-fau.jpg":
            #     image = image.replace(image, 'thin_tile/decoration.jpg');
            # if image=="broken_bridge/C:%5Cfakepath%5Cbroken+bridge.jpg":
            #     image = image.replace(image, 'broken_bridge/12.jpg');
            # if image=="crinkled_bag/?url=http%3A%2F%2Fs3.amazonaws.com%2Fassets.prod.vetstreet.com%2Fe3%2F09%2F8924d0e540dba5bb9cbd8b8a24be%2FNeko%20Nappers%20sleep%20bag-590kgs.jpg":
            #     image = image.replace(image, 'crinkled_bag/22.jpg');
            # if image=="new_town/new-town-princes-street-vanaf-edinburgh-cas(p:location,3285)(c:0).jpg":
            #     image = image.replace(image, 'new_town/new.jpg');
            # if image=="ripe_fig/?url=http%3A%2F%2Fcdn.hgtvgardens.com%2Fb4%2Ff9%2F6a69a465458f8374749512ca5cef%2FRX-DK-CGG27302_ripe-figs_s3x4.jpg":
            #     image = image.replace(image, 'ripe_fig/3A.jpg');
            # if image=="tiny_wave/%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%8A%E0%B8%B2%E0%B8%82%E0%B8%B2%E0%B8%95%E0%B8%B4%E0%B8%98%E0%B8%B8%E0%B8%A3%E0%B8%81%E0%B8%B4%E0%B8%88+-+%E0%B8%AB%E0%B8%B8%E0%B9%88%E0%B8%99%E0%B8%A2%E0%B8%99%E0%B8%95%E0%B9%8C+Tiny+Wave.jpg":
            #     image = image.replace(image, 'tiny_wave/E.jpg');
            # if image=="crushed_bay/?url=http%3A%2F%2Fcdn.hgtvgardens.com%2F6e%2F12%2Faad9fd47472dbeda43357794ddbd%2Fteenager-stink-bug.jpg":
            #     image = image.replace(image, 'crushed_bay/com.jpg');
            # if image=="grimy_floor/image.axd?picture=2009%2F9%2Fmountaintop-removal520.jpg":
            #     image = image.replace(image, 'grimy_floor/image.jpg');
            # if image=="heavy_bag/master:EVR024.jpg":
            #     image = image.replace(image, 'heavy_bag/master.jpg');
            # if image=="modern_jewelry/IND009,%2BInnocente,%2BTough%2BGuy,%2Bdesigner%2Bcufflinks%2Baustralia,%2Bmens%2Brings,%2Bmens%2Bwedding%2Bbands,%2Bgift%2Bideas%2Bfor%2Bmen,%2Bcontemporary%2Bjewellery,%2Bmelbourne,%2Bjewellery%2Bfor%2Bmen.jpg":
            #     image = image.replace(image, 'modern_jewelry/IND009.jpg');
            # if image=="moldy_milk/safe_image.php?d=AQDqHUJZQgUZzXYB&w=720&h=540&url=http%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2Fa%2Fa6%2FLatte_025.jpg%2F720px-Latte_025.jpg":
            #     image = image.replace(image, 'moldy_milk/safe.jpg');
            # if image=="painted_building/architecture-high-class-skyscraper-apartment-building-plans-project-designed-with-medium-balcony-with-glass-fence-also-beautiful-white-painted-building-and-interesting-black-glass-main-lobby-for-exclu-615x410.jpg":
            #     image = image.replace(image, 'painted_building/architecture.jpg');
            # if image=="dull_knife/safe_image.php?d=AQCyIfPCxNl5FE16&w=720&h=540&url=http%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2Fd%2Fd0%2FDull_Knife_Battlefield_Village_in_Foreground.jpg%2F720px-Dull_Knife_Battlefield_Village_in_Foreground.jpg":
            #     image = image.replace(image, 'dull_knife/safe_image.jpg');
            # if image=="engraved_frame/warp_image.php?imagefile=warp_sites%2Foushop.g6%2Ffiles%2FWeb_Oxf_Ltd_14_020.jpg":
            #     image = image.replace(image, 'engraved_frame/warp.jpg');
            # if image=="engraved_frame/warp_image.php?imagefile=warp_sites%2Foushop.g6%2Ffiles%2FWeb_Oxf_Ltd_14_021.jpg":
            #     image = image.replace(image, 'engraved_frame/warp1.jpg');
            # if image=="grimy_bathroom/?url=http%3A%2F%2Fcdn.frontdoor.com%2F33%2Fb7%2F34a1996d4531b93b38268a02ff00%2Fhdts-2501-bathroom-mirrors-flowers.jpg":
            #     image = image.replace(image, 'grimy_bathroom/frontdoor.jpg');
            # if image=="huge_tower/resize?key=1e6a1a1efdb011df84894040444cdc60&url=http%3A%2F%2Fpbs.twimg.com%2Fmedia%2FBgS31hXIIAAFgMs.jpg":
            #     image = image.replace(image, 'huge_tower/twimg.jpg');
            # if image=="molten_stream/article-urn:publicid:ap.org:ed824ab03a954773b93dc699bec31dd9-6Tfu0Dy6r-HSK1-987_634x412.jpg":
            #     image = image.replace(image, 'molten_stream/article.jpg');
            # if image=="ruffled_shower/bathroom-decorations-beautiful-gradient-charcoal-buttonhole-ruffled-shower-curtain-with-fancy-pure-white-bathtub-in-minimalist-bathroom-design-ravishing-ruffle-shower-curtain-for-cool-bathroom-decor-948x948.jpg":
            #     image = image.replace(image, 'ruffled_shower/bathroom.jpg');
            # if image=="unripe_coffee/vietnamese-workers-processing-coffee-photo-by-vn-economy-there-are-many-opinions-that-suggest-vietnam-has-become-the-worlds-single-largest-exporter-of-robusta-coffee-is-it-right-992718-daumocca-c81f1.jpg":
            #     image = image.replace(image, 'unripe_coffee/vietnamese.jpg');
            # if image =="weathered_furniture/master:RVS1704.jpg":
            #     image = image.replace(image, 'weathered_furniture/master.jpg');

            if obj == 'Boots.Ankle' or obj == 'Boots.Knee.High' or \
                    obj == 'Boots.Mid-Calf':
                super_idx=1

            if obj == 'Sandals' or obj == 'Slippers':
                super_idx=2

            if obj == 'Shoes.Boat.Shoes' or obj == 'Shoes.Clogs.and.Mules' or \
                    obj == 'Shoes.Flats' or obj == 'Shoes.Heels' or \
                    obj == 'Shoes.Loafers' or obj == 'Shoes.Oxfords' or \
                    obj == 'Shoes.Sneakers.and.Athletic.Shoes':
                super_idx=3

            curr_data = [image, attr, obj, super_idx]
            string = ":|?"
            for i in string:
                for j in image:
                    if i == j:
                        j.replace(j, '_');

            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                # Skip incomplete pairs, unknown pairs and unknown set
                continue

            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val':
                val_data.append(curr_data)
            else:
                test_data.append(curr_data)

        return train_data, val_data, test_data

    def get_dict_data(self, data, pairs):
        data_dict = {}
        for current in pairs:
            data_dict[current] = []

        for current in data:
            image, attr, obj = current
            data_dict[(attr, obj)].append(image)

        return data_dict

    def reset_dropout(self):
        '''
        Helper function to sample new subset of data containing a subset of pairs of objs and attrs
        '''
        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Using sampling from random instead of 2 step numpy
        n_pairs = int((1 - self.pair_dropout) * len(self.train_pairs))

        self.sample_pairs = random.sample(self.train_pairs, n_pairs)
        print('Sampled new subset')
        print('Using {} pairs out of {} pairs right now'.format(
            n_pairs, len(self.train_pairs)))

        self.sample_indices = [i for i in range(len(self.data))
                               if (self.data[i][1], self.data[i][2]) in self.sample_pairs
                               ]
        print('Using {} images out of {} images right now'.format(
            len(self.sample_indices), len(self.data)))

    def sample_negative(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Returns
            Tuple of a different attribute, object indexes
        '''
        new_attr, new_obj = self.sample_pairs[np.random.choice(
            len(self.sample_pairs))]

        while new_attr == attr and new_obj == obj:
            new_attr, new_obj = self.sample_pairs[np.random.choice(
                len(self.sample_pairs))]

        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object
        '''
        new_attr = np.random.choice(self.obj_affordance[obj])

        while new_attr == attr:
            new_attr = np.random.choice(self.obj_affordance[obj])

        return self.attr2idx[new_attr]

    def sample_train_affordance(self, attr, obj):
        '''
        Inputs
            attr: String of valid attribute
            obj: String of valid object
        Return
            Idx of a different attribute for the same object from the training pairs
        '''
        new_attr = np.random.choice(self.train_obj_affordance[obj])

        while new_attr == attr:
            new_attr = np.random.choice(self.train_obj_affordance[obj])

        return self.attr2idx[new_attr]

    def generate_features(self, out_file, model):
        '''
        Inputs
            out_file: Path to save features
            model: String of extraction model
        '''
        # data = self.all_data
        data = ospj(self.root, 'images')
        files_before = glob(ospj(data, '**', '*.jpg'), recursive=True)
        files_all = files_before
        # for current in files_before:
        #     parts = current.split('/')
        #     if "cgqa" in self.root:
        #         files_all.append(parts[-1])
        #     else:
        #         files_all.append(os.path.join(parts[-2], parts[-1]))
        transform = dataset_transform('test', self.norm_family)


        image_feats = []
        image_files = []
        for chunk in tqdm(
                chunks(files_all, 512), total=len(files_all) // 512, desc=f'Extracting features {model}'):
            files = chunk
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            image_feats.append(imgs.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)

    def __getitem__(self, index):
        '''
        Call for getting samples
        '''
        index = self.sample_indices[index]

        image, attr, obj, superindex = self.data[index]

        #选择同状态的图片

        img_att_all = []
        attr1 = []
        obj1= []
        for idx in range(len(self.data)):
            if self.data[idx][1] == attr and self.data[idx][3] == superindex:
                img_att = self.data[idx][0]
                img_att_all.append(img_att)
                attr1.append(self.data[idx][1])
                obj1.append(self.data[idx][2])
        idx = random.randint(0, len(img_att_all) - 1)
        img_att = img_att_all[idx]
        attr_1= attr1[idx]
        obj_1 = obj1[idx]


        # 选择相同对象的图片
        img_obj_all = []
        attr2 = []
        obj2= []
        for idx in range(len(self.data)):
            if self.data[idx][2] == obj:
                img_obj = self.data[idx][0]
                img_obj_all.append(img_obj)
                attr2.append(self.data[idx][1])
                obj2.append(self.data[idx][2])
        idx = random.randint(0, len(img_obj_all) - 1)
        img_obj = img_obj_all[idx]
        attr_2= attr2[idx]
        obj_2 = obj2[idx]

        # 选择相同状态但是不同超类的图片
        img_obj_all_all=[]
        attr3 = []
        obj3= []
        for idx in range(len(self.data)):
            if self.data[idx][1]==attr and self.data[idx][3]!=superindex :
                img_att_super = self.data[idx][0]
                img_obj_all_all.append(img_att_super)
                attr3.append(self.data[idx][1])
                obj3.append(self.data[idx][2])
        if len(img_obj_all_all)==0:
            for idx in range(len(self.data)):
                if self.data[idx][1]!= attr and self.data[idx][3] != superindex:
                    img_obj_Super = self.data[idx][0]
                    attr_3 = self.data[idx][1]
                    obj_3 = self.data[idx][2]
                    break
        else:
            idx = random.randint(0, len(img_obj_all_all) - 1)
            img_obj_Super = img_obj_all_all[idx]
            attr_3 = attr3[idx]
            obj_3 = obj3[idx]
        # Decide what to output
        if not self.update_features:
            img = self.activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)
            img_att_1 = self.loader(img_att)
            img_att = self.transform(img_att_1)
            img_obj1 = self.loader(img_obj)
            img_obj = self.transform(img_obj1)
            img_obj_Super1 = self.loader(img_obj_Super)
            img_obj_diSuper = self.transform(img_obj_Super1)

#########################再把语义加上，首先先把语义导入进来##############################
        # self.embeddings = load_data()
        # self.embeddings = self.embeddings.to(device)
        # attrData = self.embeddings[0:len(self.attrs)]
        # objData = self.embeddings[len(self.attrs):]
        # att_embedding = attrData[self.attr2idx[attr]]
        # obj_embedding = objData[self.obj2idx[obj]]

        data = [img, img_att, img_obj, img_obj_diSuper, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)],superindex]

        if self.phase == 'train':
            all_neg_attrs = []
            all_neg_objs = []

            for curr in range(self.num_negs):
                neg_attr, neg_obj = self.sample_negative(attr, obj)  # negative for triplet lose
                all_neg_attrs.append(neg_attr)
                all_neg_objs.append(neg_obj)

            neg_attr, neg_obj = torch.LongTensor(all_neg_attrs), torch.LongTensor(all_neg_objs)

            # note here
            # if len(self.train_obj_affordance[obj]) > 1:
            #     inv_attr = self.sample_train_affordance(attr, obj)  # attribute for inverse regularizer
            # else:
            #     inv_attr = (all_neg_attrs[0])
            #
            # comm_attr = self.sample_affordance(inv_attr, obj)  # attribute for commutative regularizer

            data += [neg_attr, neg_obj]

        # Return image paths if requested as the last element of the list
        if self.return_images and self.phase != 'train':
            data.append(image)

        return data

    def __len__(self):
        '''
        Call for length
        '''
        return len(self.sample_indices)
