import copy
import random 
import math
from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, tokenize, convert_weights_prompt
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from utils.simple_tokenizer import SimpleTokenizer
from typing import Optional, Tuple, Union
from .scaler import Scaler
import torch.nn.functional as F
from .scaler import Scaler
from .Prefix import Prefix

tokenizer = SimpleTokenizer()

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)]) #nn.ModuleList 是 PyTorch 中的一个类，用于管理一组模块（modules）。代码通过列表解析的方式，创建了一个包含 N 个模块的列表，每个模块都是 module 的深拷贝。copy.deepcopy() 函数用于创建模块的深拷贝，确保每个模块在列表中是独立的副本，而不是共享相同的内存。

 
class Adapter(nn.Module):
    def __init__(self, c_in, c_out, reduction=8):
        super(Adapter, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(c_in, c_in // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(c_in // reduction, c_out, bias=False),
        #     # nn.ReLU(inplace=True)
        # )
        self.down =  nn.Linear(c_in, c_in // reduction, bias=False)
        self.ac = nn.ReLU(inplace=True)
        self.up = nn.Linear(c_in // reduction, c_out, bias=False)
        
        # self.linear = nn.Linear(c_in, c_out)

    def forward(self, x):
        res = x
        x = self.ac(self.down(x)) 
        x = self.up(x)
        # out = res + x
        # return out
        return x
    
    
class KVLoRA(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank: Union[int, Tuple[int]] = 5,
        scale: Union[None, float, Tuple[float, float]] = None,
    ):
        super().__init__()

        assert rank > 0

        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.zeros((rank, in_features))) for _ in range(2)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros((out_features, rank))) for _ in range(2)]
        )

        if not isinstance(scale, tuple):
            scale = (scale, scale)
        self.scale = nn.ModuleList([Scaler(scale[0]), Scaler(scale[1])])

        self.reset_parameters()


    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        for i in range(2):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i])

    def forward(self):
        items = zip(self.scale, self.lora_A, self.lora_B)
        weight = torch.cat([s(B @ A) for s, A, B in items], dim=0)
        return weight
 


class MultiKVLoRA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.compound_prompts_depth = cfg.depth_lora #原始：9
        self.rank = cfg.rank #原始：4
        self.lora_text = nn.ModuleList([KVLoRA(in_features=512, out_features=512, rank=self.rank, scale=1.0) for _ in range(self.compound_prompts_depth)])
        self.lora_image = nn.ModuleList([KVLoRA(in_features=768, out_features=768, rank=self.rank, scale=1.0) for _ in range(self.compound_prompts_depth)])

        
    def forward(self):
        weight_text_list = []
        weight_image_list = []
        for single_module in self.lora_text:
            weight_text_list.append(single_module())
        for single_module in self.lora_image:
            weight_image_list.append(single_module())
        return weight_text_list, weight_image_list
          

# prefix
class VitPrefixLearner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        kwargs={}
        self.prefix_len = cfg.prefix_length
        self.compound_prompts_depth = cfg.depth_prefix #原始：9

        print('MultiPrefixLearner design')
        print(f"Number of prefix (tokens): {self.prefix_len}")
        kwargs["dim"] = 768
        kwargs["length"] = self.prefix_len
        self.vit_prefix =  nn.ModuleList([Prefix(**kwargs) for i in range(cfg.depth_prefix)])
        
    def forward(self):
        return self.vit_prefix
        
class TextPrefixLearner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        kwargs={}
        self.prefix_len = cfg.prefix_length
        self.compound_prompts_depth = cfg.depth_prefix #原始：9

        print('MultiPrefixLearner design')
        print(f"Number of prefix (tokens): {self.prefix_len}")
        kwargs["dim"] = 512
        kwargs["length"] = self.prefix_len
        self.text_prefix =  nn.ModuleList([Prefix(**kwargs) for i in range(cfg.depth_prefix)])
        
    def forward(self):
        return self.text_prefix
        
class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg, state_dict = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size, args.n_ctx, (args.depth_lora,args.depth_prefix,args.depth_adapter))
        self.embed_dim = base_cfg['embed_dim']

        # prefix
        self.vit_prefix_learner = VitPrefixLearner(args)
        self.text_prefix_learner = TextPrefixLearner(args)
        
        
        self.apply(self.init_weights) # random init must before loading pretrain
        self.LORA_learner = MultiKVLoRA(args)
        self.base_model.load_param(state_dict)
           
        # covert model to fp16
        if torch.cuda.is_available():
            convert_weights(self.base_model)

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
            
        for i in range(args.depth_adapter, 12):
            nn.init.kaiming_uniform_(self.base_model.visual.transformer.resblocks[i].adapter_ln_mlp.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].adapter_ln_mlp.down.bias)
            nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].adapter_ln_mlp.up.weight)
            nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].adapter_ln_mlp.up.bias)
            
            nn.init.kaiming_uniform_(self.base_model.visual.transformer.resblocks[i].adapter_ln_msa.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].adapter_ln_msa.down.bias)
            nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].adapter_ln_msa.up.weight)
            nn.init.zeros_(self.base_model.visual.transformer.resblocks[i].adapter_ln_msa.up.bias)
            
        for i in range(args.depth_adapter, 12):
            nn.init.kaiming_uniform_(self.base_model.transformer.resblocks[i].adapter_ln_msa.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.base_model.transformer.resblocks[i].adapter_ln_msa.down.bias)
            nn.init.zeros_(self.base_model.transformer.resblocks[i].adapter_ln_msa.up.weight)
            nn.init.zeros_(self.base_model.transformer.resblocks[i].adapter_ln_msa.up.bias)
            
            nn.init.kaiming_uniform_(self.base_model.transformer.resblocks[i].adapter_ln_mlp.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.base_model.transformer.resblocks[i].adapter_ln_mlp.down.bias)
            nn.init.zeros_(self.base_model.transformer.resblocks[i].adapter_ln_mlp.up.weight)
            nn.init.zeros_(self.base_model.transformer.resblocks[i].adapter_ln_mlp.up.bias)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        # print(module)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
         
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q.float()),
                self.ln_pre_i(k.float()),
                self.ln_pre_i(v.float()),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        
        # prefix
        vit_prefix_list = self.vit_prefix_learner()
        text_prefix_list = self.text_prefix_learner()

        if self.args.depth_prefix > 0:
            for i in range(self.args.depth_prefix):
                self.base_model.visual.transformer.resblocks[i].attn.attach_prefix(vit_prefix_list[i])
                self.base_model.transformer.resblocks[i].attn.attach_prefix(text_prefix_list[i])

        weight_text_list, weight_image_list = self.LORA_learner()
        image_feats, text_feats = self.base_model(images, caption_ids, weight_text_list, weight_image_list, self.args)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        
        # image_feats_clip = self.base_model.encode_image_clip(images)

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
            
        if 'triplet' in self.current_task:
            ret.update({'triplet_loss':objectives.compute_triplet(i_feats, t_feats)})
        
        if 'id' in self.current_task:
            # image_logits = self.classifier(i_feats.half()).float()
            # text_logits = self.classifier(t_feats.half()).float()
            image_logits = self.classifier(i_feats.float()).float()
            text_logits = self.classifier(t_feats.float()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
            
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids, weight_text_list, self.args.prefix_length, (self.args.depth_lora,  self.args.depth_prefix,  self.args.depth_adapter))

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
        

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    
    return model
