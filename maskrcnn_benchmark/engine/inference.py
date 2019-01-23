# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import socket
import tempfile
import time
from collections import OrderedDict
from datetime import datetime as dt

import torch
from tqdm import tqdm

from maskrcnn_benchmark.kitti_vis import vis_2d_boxes_list, read_img
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from ..structures.bounding_box import BoxList
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict

def prepare_for_bbox3d_detection(predictions, dataset, output_folder):
    count = 0
    for image_id, prediction in enumerate(predictions):
        # TODO image_id is what
        count = count + 1
        print(count)
        idx = dataset.id_to_img_map[image_id]
        original_id = dataset.image_name[idx]
        if len(prediction) == 0:
            continue

        image_size = dataset.get_img_info(image_id)

        image_width = image_size["width"]
        image_height = image_size["height"]

        prediction = prediction.convert("xyxy")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        boxes_3d = prediction.get_field("boxes_3d").bbox_3d.tolist()
        alphas = prediction.get_field("alphas").tolist()

        save_kitti_3d_result(boxes, original_id, scores, output_folder, dataset, alphas, boxes_3d)

    return output_folder

def save_kitti_3d_result(box, original_id, scores, output_folder, dataset, alphas, boxes_3d):

    if output_folder:
        output_folder = os.path.join(output_folder, 'detections', 'data')
        if not os.path.exists(output_folder):
            mkdir(output_folder)
    filename = os.path.join(output_folder, original_id + ".txt")

    with open(filename, 'wt') as f:
        if len(box) == 0:
            return
        for k in range(len(box)):
            height = box[k][3] - box[k][1] + 1
            if height < 25:
                continue
            ry, h, w, l, tx, ty, tz = boxes_3d[k]
            alpha = alphas[k]

            f.write(
                '{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n' \
                    .format("Car", \
                            alpha, box[k][0], box[k][1], box[k][2], box[k][3], \
                            h, w, l, tx, ty, tz, ry, scores[k]))


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.deprecated.get_world_size()
        if torch.distributed.deprecated.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} images".format(len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    # config MODEL WEIGHT path expect like that
    # "/raid/kitti3doutput/e2e_mask_rcnn_R_50_FPN_1x/Dec20-12-13-59_DGX-1-A7_step/model_0002500.pth"
    cfg_name = cfg.MODEL.WEIGHT.split('/')[-3]
    model_step = cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0]
    output_folder = os.path.join(output_folder, cfg_name, model_step)
    if not os.path.exists(output_folder):
        mkdir(output_folder)

    # if output_folder:
    #     torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format")
    coco_results = {}
    # if "bbox" in iou_types:
    #     logger.info("Preparing bbox results")
    #     coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset, output_folder)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)
    if "bbox3d" in iou_types:
        logger.info("Preparing bbox3d results")
        coco_results["bbox3d"] = prepare_for_bbox3d_detection(predictions, dataset, output_folder)
        return

    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            res = evaluate_predictions_on_coco(
                dataset.coco, coco_results[iou_type], file_path, iou_type
            )
            results.update(res)
    logger.info(results)
    check_expected_results(results, expected_results, expected_results_sigma_tol)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "coco_results.pth"))

    return results, coco_results, predictions

