import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from .coco_utils import results2json, fast_eval_recall, orientation_results2json
from .mean_ap import eval_map
from .eval_mr import COCOeval as COCOMReval
from mmdet import datasets

import json
from .bbox_overlaps import bbox_overlaps
from sklearn.metrics import classification_report


class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = [] if self.dataset.with_crowd else None
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                labels = np.concatenate([labels, ann['labels_ignore']])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 interval=1,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(
            dataset, interval=interval)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        res_types = ['bbox', 'segm'] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            cocoDt = cocoGt.loadRes(result_files[res_type])
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        for res_type in res_types:
            os.remove(result_files[res_type])


class CocoDistEvalMRHook(DistEvalHook):
    """ EvalHook for MR evaluation.

    Args:
        res_types(list): detection type, currently support 'bbox'
            and 'vis_bbox'.
    """
    def __init__(self, dataset, interval=1, res_types=['bbox']):
        super().__init__(dataset, interval)
        self.res_types = res_types

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = results2json(self.dataset, results, tmp_file)

        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        for res_type in self.res_types:
            assert res_type in ['bbox', 'vis_bbox']
            try:
                cocoDt = cocoGt.loadRes(result_files['bbox'])
            except IndexError:
                print('No prediction found.')
                break
            metrics = ['MR_Reasonable', 'MR_Small', 'MR_Middle', 'MR_Large',
                       'MR_Bare', 'MR_Partial', 'MR_Heavy', 'MR_R+HO']
            cocoEval = COCOMReval(cocoGt, cocoDt, res_type)
            cocoEval.params.imgIds = imgIds
            for id_setup in range(0,8):
                cocoEval.evaluate(id_setup)
                cocoEval.accumulate()
                cocoEval.summarize(id_setup)
                
                key = '{}'.format(metrics[id_setup])
                val = float('{:.3f}'.format(cocoEval.stats[id_setup]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_MR_copypaste'.format(res_type)] = (
                '{mr[0]:.3f} {mr[1]:.3f} {mr[2]:.3f} {mr[3]:.3f} '
                '{mr[4]:.3f} {mr[5]:.3f} {mr[6]:.3f} {mr[7]:.3f} ').format(
                    mr=cocoEval.stats[:8])
        runner.log_buffer.ready = True
        os.remove(result_files['bbox'])


def xywh2xyxy(bbox):
        return [
            bbox[0],
            bbox[1],
            bbox[0] + bbox[2] - 1,
            bbox[1] + bbox[3] - 1,
        ]

class OrientationCocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0')
        result_files = orientation_results2json(self.dataset, results, tmp_file)

        res_types = ['bbox', 'segm'] if runner.model.module.with_mask else ['bbox']
        
        cocoGt = self.dataset.coco
        imgIds = cocoGt.getImgIds()
        
        # gt_ann_ids = cocoGt.getAnnIds(imgIds=imgIds)
        # gt_ann_info = cocoGt.loadAnns(gt_ann_ids)
        # gt_ann = self.dataset._parse_ann_info(ann_info, with_mask=self.dataset.with_mask, with_orientation=True)

        with open(result_files['bbox']) as f:
            bbox_result = json.load(f)

        pred_anns = []
        gt_anns = []
        for i, res in enumerate(bbox_result):
            if res['score'] >= 0.5:
                gt_ann_ids = cocoGt.getAnnIds(imgIds=[res['image_id']])
                gt_ann_info = cocoGt.loadAnns(gt_ann_ids)
                gt_ann = self.dataset._parse_ann_info(gt_ann_info, with_mask=self.dataset.with_mask, with_orientation=True)
                pred_bbox = np.array([xywh2xyxy(res['bbox'])])
                gt_bbox = np.array([gt_bbox for gt_bbox in gt_ann['bboxes']])
                max_idx = np.argmax(bbox_overlaps(pred_bbox, gt_bbox))
                gt_anns.append(gt_ann['orientation_labels'][max_idx])
                pred_anns.append(res['orientation'])

        if pred_anns:
            runner.log_buffer.output['classification_report'] = classification_report(gt_anns, pred_anns)
            print(classification_report(gt_anns, pred_anns))
            runner.log_buffer.ready = True
        else:
            print("No positive bboxes")

        for res_type in res_types:
            cocoDt = cocoGt.loadRes(result_files[res_type])
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        for res_type in res_types:
            os.remove(result_files[res_type])
