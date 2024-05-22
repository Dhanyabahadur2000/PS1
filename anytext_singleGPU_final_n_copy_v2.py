import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import einops
import numpy as np
import torch
import random
from PIL import ImageFont

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from t3_dataset import draw_glyph, draw_glyph2, draw_glyph1, get_caption_pos, draw_glyph_z, draw_glyph10
from dataset_util import load
from tqdm import tqdm
import argparse
import time
import torchvision.transforms as T
import torch
import numpy

save_memory = False
# parameters
config_yaml = './models_yaml/anytext_sd15.yaml'
ckpt_path = '/home/ubuntu/.cache/modelscope/hub/damo/cv_anytext_text_generation_editing/anytext_v1.1.ckpt'
json_path = '/home/ubuntu/anytext_v1/AnyText/data/test1k_1.json'
output_dir = '/home/ubuntu/anytext_v1/AnyText/eval/gen_imgs_test_1'
num_samples = 1
image_resolution = 512
strength = 1.0
ddim_steps = 20
scale = 9.0
seed = 100
eta = 0.0
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, watermark'
PLACE_HOLDER = '*'
max_chars = 20
max_lines = 20
font = ImageFont.truetype('./font/Arial_Unicode.ttf', size=60)


def parse_args():
    parser = argparse.ArgumentParser(description='generate images')
    parser.add_argument('--input_json', type=str, default=json_path)
    parser.add_argument('--output_dir', type=str, default=output_dir)
    parser.add_argument('--ckpt_path', type=str, default=ckpt_path)
    args = parser.parse_args()
    return args


def arr2tensor(arr, bs):
    arr = np.transpose(arr, (2, 0, 1))
    _arr = torch.from_numpy(arr.copy()).float().cuda()
    _arr = torch.stack([_arr for _ in range(bs)], dim=0)
    return _arr


def load_data(input_path):
    content = load(input_path)
    d = []
    count = 0
    for gt in content['data_list']:
        info = {}
        info['img_name'] = gt['img_name']
        info['caption'] = gt['caption']
        if PLACE_HOLDER in info['caption']:
            count += 1
            info['caption'] = info['caption'].replace(PLACE_HOLDER, " ")
        if 'annotations' in gt:
            polygons = []
            texts = []
            pos = []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0:
                    continue
                if annotation['valid'] is False:
                    continue
                polygons.append(annotation['polygon'])
                texts.append(annotation['text'])
                pos.append(annotation['pos'])
            info['polygons'] = [np.array(i) for i in polygons]
            info['texts'] = texts
            info['pos'] = pos
        d.append(info)
    print(f'{input_path} loaded, imgs={len(d)}')
    if count > 0:
        print(f"Found {count} image's caption contain placeholder: {PLACE_HOLDER}, change to ' '...")
    return d


def draw_pos1(ploygon, prob=1.0, shape1 = (512, 512, 1)):
    img = np.zeros(shape1)
    if random.random() < prob:
        pts = ploygon.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=255)
    return img/255.


def draw_pos(ploygon, prob=1.0, shape1 = (512, 512, 1)):
    img = np.zeros(shape1)
    if random.random() < prob:
        pts = ploygon.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=255)
    return img/255.


def get_item(data_list, item):
    item_dict = {}
    cur_item = data_list[item]
    item_dict['img_name'] = cur_item['img_name']
    item_dict['caption'] = cur_item['caption']
    item_dict['glyphs'] = []
    item_dict['glyphs1'] = []
    item_dict['gly_line'] = []
    item_dict['positions'] = []
    item_dict['texts'] = []
    texts = cur_item.get('texts', [])
    if len(texts) > 0:
        sel_idxs = [i for i in range(len(texts))]
        if len(texts) > max_lines:
            sel_idxs = sel_idxs[:max_lines]
        pos_idxs = [cur_item['pos'][i] for i in sel_idxs]
        item_dict['caption'] = get_caption_pos(item_dict['caption'], pos_idxs, 0.0, PLACE_HOLDER)
        item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
        item_dict['texts'] = [cur_item['texts'][i][:max_chars] for i in sel_idxs]
        print("length of item_dict['texts'] :- ", len(item_dict['texts']))
        print("length of item_dict['polygons'] :- ", len(item_dict['polygons']))
        # glyphs
        for idx, text in enumerate(item_dict['texts']):
            print("text :- ", text)
            gly_line = draw_glyph(font, text)
            # no rotation
            glyphs = draw_glyph2(font, text, item_dict['polygons'][idx], scale=2)
            # with rotation
            glyphs1, font_size1 = draw_glyph1(font, text, item_dict['polygons'][idx], scale=2)
            item_dict['glyphs'] += [glyphs]
            item_dict['glyphs1'] += [glyphs1]
            item_dict['gly_line'] += [gly_line]
        # mask_pos
        for polygon in item_dict['polygons']:
            item_dict['positions'] += [draw_pos(polygon, 1.0)]




        ######################
        for text2, polygon2 in zip(item_dict['texts'], item_dict['polygons']):
            print("text2 :- ", text2)
            print("type of polygon2 :- ", type(polygon2))
            print(" polygon2 :- ", polygon2)


            max_x = -1024
            max_y = -1024
            min_x = 1024
            min_y = 1024

            for array in polygon2:
                print(array)
                # for xy in array:
                #     print(xy)
                if(array[0]>max_x):
                    max_x=array[0]
                if(array[0]<min_x):
                    min_x=array[0]
                if(array[1]<min_y):
                    min_y=array[1]
                if(array[1]>max_y):
                    max_y = array[1]
            print("max and min of coords :- ", max_x, max_y, min_x, min_y)
            Height = (max_y - min_y)
            Width = (max_x - min_x)

            try:
                transform = T.ToPILImage()
                # with rotation
                glyphs2, font_size2 = draw_glyph10(font, text2, polygon2, scale=1, 
                                      width = 512, height = 512#width = store_size[0]+100, height=store_size[1]+100
                                      )
                print("type of glyphs2 :- ", type(glyphs2))
                print("shape of glyphs2 :- ", glyphs2.size)
                glyphs2 = glyphs2[min_y-10:max_y+10, min_x-10:max_x+10]
                glyphs2 = arr2tensor(glyphs2, 1)
                glyphs2 = glyphs2.squeeze()
                glyphs2 = transform(glyphs2)


                # shape of glyph image is different from mask and rotated image
                gly_line2 = draw_glyph_z(font, text2, font_size2)
                print("type of gly_line2 :- ", type(gly_line2))
                print("shape of gly_line2 :- ", gly_line2.size)
                #gly_line2 = gly_line2[min_y:max_y, min_x:max_x]
                print("type of gly_line2 :- ", type(gly_line2))
                gly_line2 = arr2tensor(gly_line2, 1)
                print("type of gly_line2 :- ", type(gly_line2))
                gly_line2 = gly_line2.squeeze()
                print("type of gly_line2 :- ", type(gly_line2))
                gly_line2 = transform(gly_line2)
                gly_line2 = gly_line2.resize((Width+20, Height+20))
                print("type of gly_line2 :- ", type(gly_line2))
                store_size = gly_line2.size

            


                mask_pos2 = draw_pos1(polygon2, 1.0, 
                                     (512, 512, 1),
                                     )
                print("type of mask_pos2 :- ", type(mask_pos2))
                print("shape of mask_pos2 :- ", mask_pos2.size)
                mask_pos2 = mask_pos2[min_y-10:max_y+10, min_x-10:max_x+10]
                mask_pos2 = arr2tensor(mask_pos2, 1)
                mask_pos2 = mask_pos2.squeeze()
                mask_pos2 = transform(mask_pos2)



                name1 = "glyph_image"
                name2 = "pos_image"
                name3 = "rotated_glyph_image"
                index = 0
                directory = '/home/ubuntu/anytext_v1/AnyText/eval/training_data_v4'
                while True:
                    filename =  f"{name1}_{index}.jpg"
                    filename1 = f"{name2}_{index}.jpg"
                    filename2 = f"{name3}_{index}.jpg"
                    file_path = os.path.join(directory, filename)
                    file_path1 = os.path.join(directory, filename1)
                    file_path2 = os.path.join(directory, filename2)
                    if not (os.path.exists(file_path) 
                        or (os.path.exists(file_path1)
                        or (os.path.exists(file_path2)))
                        ):

                        break

                    index += 1
            #output_file_path = os.path.join(output_dir, "final_image.jpg")
                    # glyph_image.save(file_path)
                    # pos_image.save(file_path1)
                    # rotated_glyph_image.save(file_path2)



                gly_line2.save(file_path)
                mask_pos2.save(file_path1)
                glyphs2.save(file_path2)
            except:
                continue






    fill_caption = False
    if fill_caption:  # if using embedding_manager, DO NOT fill caption!
        for i in range(len(item_dict['texts'])):
            r_txt = item_dict['texts'][i]
            item_dict['caption'] = item_dict['caption'].replace(PLACE_HOLDER, f'"{r_txt}"', 1)
    # padding
    n_lines = min(len(texts), max_lines)
    item_dict['n_lines'] = n_lines
    n_pad = max_lines - n_lines
    if n_pad > 0:
        item_dict['glyphs'] += [np.zeros((512*2, 512*2, 1))] * n_pad
        item_dict['glyphs1'] += [np.zeros((512*2, 512*2, 1))] * n_pad
        item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
        item_dict['positions'] += [np.zeros((512, 512, 1))] * n_pad
    return item_dict

import random
random.seed(42)

def process(model, ddim_sampler, item_dict, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta):
    with torch.no_grad():
        prompt = item_dict['caption']
        n_lines = item_dict['n_lines']
        pos_imgs = item_dict['positions']
        glyphs = item_dict['glyphs']
        glyphs1 = item_dict['glyphs1']
        gly_line = item_dict['gly_line']
        hint = np.sum(pos_imgs, axis=0).clip(0, 1)
        H, W, = (512, 512)
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        if save_memory:
            model.low_vram_shift(is_diffusing=False)
        info = {}
        info['glyphs'] = []
        info['glyphs1'] = []
        info['gly_line'] = []
        info['positions'] = []
        info['n_lines'] = [n_lines]*num_samples
        print("n_lines :- ", n_lines)
        for i in range(n_lines):
            glyph = glyphs[i]
            glyph1 = glyphs1[i]
            pos = pos_imgs[i]
            gline = gly_line[i]
            info['glyphs'] += [arr2tensor(glyph, num_samples)]
            info['glyphs1'] += [arr2tensor(glyph1, num_samples)]
            info['gly_line'] += [arr2tensor(gline, num_samples)]
            info['positions'] += [arr2tensor(pos, num_samples)]


            ######
            # transform = T.ToPILImage()
            # count = 0
            # final_image = torch.zeros(1024, 1024).to("cuda:0")
            # final_image1 = torch.zeros(1024, 1024).to("cuda:0")
            # final_image2 = torch.zeros(512, 512).to("cuda:0")
            # for indie, dataz  in enumerate(zip([arr2tensor(glyph, num_samples)], [arr2tensor(glyph1, num_samples)],
            #                                [arr2tensor(pos, num_samples)])):
            #     i, j, k = dataz
            #     for tasty, data1 in enumerate(zip(i, j, k)):
            #         text1, glyph1, position = data1
            #         #print("text1 :- ", text1)
            # #for text1, glyph1, position in (i, j, k):
            #         text1 = text1.squeeze()
            #         glyph1 = glyph1.squeeze()
            #         position = position.squeeze()
            #         print("text1 shape :- ", text1.shape)
            #         print("glyph1 shape :- ", glyph1.shape)
            #         print("position shape :- ", position.shape)
            #         text_img = transform(text1)
            #         output_file_path = os.path.join(output_dir, "glyphs_image_"+ str(count)+".jpg")
            #         text_img.save(output_file_path)

            #         glyph1_img = transform(glyph1)
            #         output_file_path1 = os.path.join(output_dir, "rotated_glyphs_image_" + str(count)+".jpg")
            #         glyph1_img.save(output_file_path1)

            #         position_img = transform(position)
            #         output_file_path2 = os.path.join(output_dir, "pos_image_" + str(count)+".jpg")
            #         position_img.save(output_file_path2)

            #         count += 1

            #         final_image = torch.add(final_image, text1)
            #         final_image1 = torch.add(final_image1, glyph1)
            #         final_image2 = torch.add(final_image2, position)

            #         glyph_image = transform(final_image)
            #         rotated_glyph_image = transform(final_image1)
            #         pos_image = transform(final_image2)

            #         name1 = "glyph_image"
            #         name2 = "pos_image"
            #         name3 = "rotated_glyph_image"
            #         index = 0
            #         directory = '/home/ubuntu/anytext_v1/AnyText/eval/training_data_v3'
            #         while True:
            #             filename =  f"{name1}_{index}.jpg"
            #             filename1 = f"{name2}_{index}.jpg"
            #             filename2 = f"{name3}_{index}.jpg"
            #             file_path = os.path.join(directory, filename)
            #             file_path1 = os.path.join(directory, filename1)
            #             file_path2 = os.path.join(directory, filename2)
            #             if not (os.path.exists(file_path) 
            #                 or (os.path.exists(file_path1)
            #                 or (os.path.exists(file_path2)))
            #                 ):

            #                 break

            #             index += 1
            # #output_file_path = os.path.join(output_dir, "final_image.jpg")
            #         glyph_image.save(file_path)
            #         pos_image.save(file_path1)
            #         rotated_glyph_image.save(file_path2)
            


        print("Image Saved")
        #####

        # get masked_x
        ref_img = np.ones((H, W, 3)) * 127.5
        masked_img = ((ref_img.astype(np.float32) / 127.5) - 1.0)*(1-hint)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().cuda()
        encoder_posterior = model.encode_first_stage(masked_img[None, ...])
        masked_x = model.get_first_stage_encoding(encoder_posterior).detach()
        info['masked_x'] = torch.cat([masked_x for _ in range(num_samples)], dim=0)

        hint = arr2tensor(hint, num_samples)

        cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[prompt + ', ' + a_prompt] * num_samples], text_info=info))
        un_cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[n_prompt] * num_samples], text_info=info))
        shape = (4, H // 8, W // 8)
        if save_memory:
            model.low_vram_shift(is_diffusing=True)
        model.control_scales = ([strength] * 13)
        tic = time.time()
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        cost = (time.time() - tic)*1000.
        if save_memory:
            model.low_vram_shift(is_diffusing=False)
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        results += [cost]
    return results


if __name__ == '__main__':
    args = parse_args()
    total = 21
    times = []
    data_list = load_data(args.input_json)
    model = create_model(config_yaml).cuda()
    model.load_state_dict(load_state_dict(args.ckpt_path, location='cuda'), strict=True)
    ddim_sampler = DDIMSampler(model)
    for i in tqdm(range(len(data_list)), desc='generator'):
        item_dict = get_item(data_list, i)
        img_name = item_dict['img_name'].split('.')[0] + '_3.jpg'
        if os.path.exists(os.path.join(args.output_dir, img_name)):
            continue
        results = process(model, ddim_sampler, item_dict, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, strength, scale, seed, eta)
        times += [results.pop()]
        if i == total:
            print(times)
            times = times[1:]
            print(f'{np.mean(times)}')
        for idx, img in enumerate(results):
            img_name = item_dict['img_name'].split('.')[0]+f'_{idx}' + '.jpg'
            cv2.imwrite(os.path.join(args.output_dir, img_name), img[..., ::-1])