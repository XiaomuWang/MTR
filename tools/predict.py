# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import argparse
import torch
from mtr.config import cfg, cfg_from_yaml_file
from mtr.models import model as model_utils
from mtr.datasets import build_dataloader
from eval_utils import eval_utils
from pathlib import Path
import datetime
from mtr.utils import common_utils
import json
import copy
import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser(description="MTR Inference")
    parser.add_argument("--cfg_file", default="/workspace/wangs/agent/MTR/output/waymo/mtr+100_percent_data/full_waymo_8/mtr+100_percent_data.yaml", type=str, help="Path to the config file")
    parser.add_argument("--ckpt", default="/workspace/wangs/agent/MTR/output/waymo/mtr+100_percent_data/full_waymo_8/ckpt/best_model.pth", type=str, help="Path to the model checkpoint")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load configuration
    cfg_from_yaml_file(args.cfg_file,cfg)

    eval_output_dir = Path('/workspace/wangs/agent/MTR/output/my_pred_exp/pred')
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir/ ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    output_file = eval_output_dir / "predictions.json"
    # Build the model
    model = model_utils.MotionTransformer(config=cfg.MODEL)
    model.load_params_from_file(args.ckpt, logger=logger)  # Load the model checkpoint

    # Prepare data loader for inference
    _, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=1,  # Adjust batch size as needed
        training=False,
        dist=False, workers=1, logger=logger
    )

    model.cuda()  
    model.eval()
    pred_dicts = []
    with torch.no_grad():
        for _, batch_dict in enumerate(test_loader):
            # tmp_batch_dict = copy.deepcopy(batch_dict)
            # center_objects_world
            # print(batch_dict)
            batch_pred_dicts = model(batch_dict)
            final_pred_list_dicts = test_loader.dataset.generate_prediction_dicts(batch_pred_dicts) # 长度与batch size有关
            for i in range(batch_dict['batch_size']):
                # 要预测的车辆数
                for j in range(len(final_pred_list_dicts[i])):
                    print(final_pred_list_dicts[i][j]['pred_trajs'].shape)
                    print(final_pred_list_dicts[i][j]['gt_trajs'].shape)
                    pred_scores = final_pred_list_dicts[i][j]['pred_scores']
                    plt.figure(figsize=(8, 6))
                    # 绘制预测轨迹
                    pred_trajs= final_pred_list_dicts[i][j]['pred_trajs']
                    x_coords = pred_trajs[:, :, 0]  # 所有轨迹的x坐标
                    y_coords = pred_trajs[:, :, 1]  # 所有轨迹的y坐标
                    # 绘制每个轨迹的点
                    for q in range(len(pred_trajs)):
                        plt.scatter(x_coords[q], y_coords[q], label=f'Trajectory {q+1} score:{pred_scores[q]:.2f}')
                        # 绘制连接点的线
                        plt.plot(x_coords[q], y_coords[q])
                    gt_trajs= final_pred_list_dicts[i][j]['gt_trajs']
                    x_gt_coords = gt_trajs[:, 0]  # 所有轨迹的x坐标
                    y_gt_coords = gt_trajs[:, 1]  # 所有轨迹的y坐标
                    # plt.savefig('scatter_plot_pred.png')
                    # 绘制真实轨迹
                    # 绘制每个轨迹的点

                    plt.scatter(x_gt_coords, y_gt_coords, label='gt_Trajectory', color='red')
                    # 绘制连接点的线
                    plt.plot(x_gt_coords, y_gt_coords)
                    # 添加图例
                    plt.legend()
                    # 设置坐标轴标签
                    plt.xlabel('X Coordinate')
                    plt.ylabel('Y Coordinate')
                    # 保存图像为文件（可以选择不同的图像格式，如PNG、JPG等）
                    plt.savefig('scatter_plot_pred_gt.png')
                    print("--------------------------------------")
            # pred_dicts += final_pred_list_dicts
            # 预测的轨迹位置 pre_trajs 拿出来 用下一帧替代 要预测的位置 然后 继续推理
            # plt.plot(trajs[:, 0], trajs[:, 1], line_type, color=(0, 1, 0), alpha=1)
            # print("------------")
            # with open(output_file, "w") as f:

            #     json.dump(pred_dicts, f)
        # You can access the predictions for further processing

    # You can save the predictions to a file or process them as needed
    # For example, you can save the predictions to a JSON file:
    # output_dir = Path("path/to/output_directory")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # output_file = output_dir / "predictions.json"
    # with open(output_file, "w") as f:
    #     import json
    #     json.dump(pred_dicts, f)

if __name__ == "__main__":
    main()
