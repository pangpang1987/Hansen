"""
使用 iter.json 中的最佳优化结果更新 input.json

从 iter.json 读取优化后的风阻（r_optimized）和风机压力（h_optimized），
更新 input.json 中对应的 r0 和 h0 值。
"""

import json
import os
import sys
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json(filepath: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"文件不存在: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析错误: {e}")
        sys.exit(1)


def save_json(data: Dict[str, Any], filepath: str, backup: bool = True):
    """保存 JSON 文件，可选择备份原文件"""
    try:
        if backup and os.path.exists(filepath):
            backup_path = filepath + '.backup'
            logger.info(f"备份原文件到: {backup_path}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"备份文件失败: {e}")
        
        # 尝试修改文件权限（如果需要）
        if os.path.exists(filepath):
            try:
                os.chmod(filepath, 0o644)  # 添加写权限
            except Exception as e:
                logger.warning(f"无法修改文件权限: {e}")
        
        # 尝试写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"已保存更新后的文件: {filepath}")
    except PermissionError:
        logger.error(f"权限不足，无法写入文件: {filepath}")
        logger.info("尝试保存到新文件...")
        # 尝试保存到新文件
        new_path = filepath + '.updated'
        try:
            with open(new_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"✓ 已保存到: {new_path}")
            logger.info(f"  请手动将 {new_path} 重命名为 {filepath}")
            logger.info(f"  或使用命令: mv {new_path} {filepath}")
        except Exception as e:
            logger.error(f"保存到新文件也失败: {e}")
            raise


def update_input_from_iter(iter_path: str = "iter.json", 
                           input_path: str = "input.json",
                           backup: bool = True):
    """
    使用 iter.json 的最佳结果更新 input.json
    
    参数:
    - iter_path: iter.json 文件路径
    - input_path: input.json 文件路径
    - backup: 是否备份原 input.json 文件
    """
    logger.info("="*70)
    logger.info("使用 iter.json 更新 input.json")
    logger.info("="*70)
    
    # 1. 加载文件
    logger.info(f"\n[1] 加载文件...")
    logger.info(f"  读取: {iter_path}")
    iter_data = load_json(iter_path)
    
    logger.info(f"  读取: {input_path}")
    input_data = load_json(input_path)
    
    # 2. 提取优化结果
    logger.info(f"\n[2] 提取优化结果...")
    
    optimized_R = iter_data.get('optimized_R', {})
    optimized_H = iter_data.get('optimized_H', {})
    
    if not optimized_R:
        logger.warning("iter.json 中未找到 optimized_R 字段")
        return
    
    logger.info(f"  找到 {len(optimized_R)} 个优化后的风阻值")
    if optimized_H:
        logger.info(f"  找到 {len(optimized_H)} 个优化后的风机压力值")
    
    # 3. 更新风阻值 (roads)
    logger.info(f"\n[3] 更新风阻值 (roads)...")
    roads = input_data.get('roads', [])
    updated_roads = 0
    not_found_roads = []
    
    # 创建风阻查找字典
    r_dict = {road['id']: i for i, road in enumerate(roads)}
    
    for road_id, r_data in optimized_R.items():
        if road_id in r_dict:
            idx = r_dict[road_id]
            old_r0 = roads[idx].get('r0', 0)
            new_r0 = r_data.get('r_optimized', old_r0)
            
            if new_r0 != old_r0:
                roads[idx]['r0'] = new_r0
                updated_roads += 1
                
                if updated_roads <= 5:  # 只显示前5个
                    change_pct = r_data.get('change_percent', 0)
                    logger.info(f"  {road_id}: {old_r0:.6e} -> {new_r0:.6e} ({change_pct:+.2f}%)")
        else:
            not_found_roads.append(road_id)
    
    logger.info(f"  已更新 {updated_roads} 个风阻值")
    if not_found_roads:
        logger.warning(f"  未在 input.json 中找到以下风路: {not_found_roads[:10]}...")
    
    # 4. 更新风机压力值 (fanHs)
    logger.info(f"\n[4] 更新风机压力值 (fanHs)...")
    fan_hs = input_data.get('fanHs', [])
    updated_fans = 0
    not_found_fans = []
    
    if optimized_H:
        # 创建风机查找字典（通过 id）
        fan_dict = {fan['id']: i for i, fan in enumerate(fan_hs)}
        
        for fan_id, h_data in optimized_H.items():
            if fan_id in fan_dict:
                idx = fan_dict[fan_id]
                old_h0 = fan_hs[idx].get('h0', 0)
                new_h0 = h_data.get('h_optimized', old_h0)
                
                if new_h0 != old_h0:
                    fan_hs[idx]['h0'] = new_h0
                    # 同时更新 h1 和 h2（如果存在）
                    if 'h1' in fan_hs[idx]:
                        fan_hs[idx]['h1'] = new_h0
                    if 'h2' in fan_hs[idx]:
                        fan_hs[idx]['h2'] = new_h0
                    
                    updated_fans += 1
                    
                    if updated_fans <= 5:  # 只显示前5个
                        change_pct = h_data.get('change_percent', 0)
                        logger.info(f"  {fan_id}: {old_h0:.2f} -> {new_h0:.2f} ({change_pct:+.2f}%)")
            else:
                not_found_fans.append(fan_id)
        
        logger.info(f"  已更新 {updated_fans} 个风机压力值")
        if not_found_fans:
            logger.warning(f"  未在 input.json 中找到以下风机: {not_found_fans[:10]}...")
    else:
        logger.info("  iter.json 中未包含 optimized_H，跳过风机压力更新")
    
    # 5. 统计信息
    logger.info(f"\n[5] 更新统计:")
    logger.info(f"  风阻更新: {updated_roads}/{len(optimized_R)}")
    if optimized_H:
        logger.info(f"  风机压力更新: {updated_fans}/{len(optimized_H)}")
    
    # 6. 保存更新后的文件
    logger.info(f"\n[6] 保存更新后的文件...")
    save_json(input_data, input_path, backup=backup)
    
    logger.info("\n" + "="*70)
    logger.info("更新完成！")
    logger.info("="*70)
    
    # 显示优化指标
    metrics = iter_data.get('metrics', {})
    if metrics:
        logger.info(f"\n优化指标:")
        logger.info(f"  最佳损失: {metrics.get('best_loss', 'N/A'):.6e}")
        logger.info(f"  迭代次数: {iter_data.get('iteration', 'N/A')}")
        logger.info(f"  函数评估: {metrics.get('eval_count', 'N/A')}")
        logger.info(f"  运行时间: {metrics.get('elapsed_time', 0):.1f} 秒")
        logger.info(f"  RMSE: {metrics.get('rmse', 'N/A'):.2f}")
        logger.info(f"  MAE: {metrics.get('mae', 'N/A'):.2f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用 iter.json 的最佳结果更新 input.json'
    )
    parser.add_argument(
        '--iter', 
        type=str, 
        default='iter.json',
        help='iter.json 文件路径（默认: iter.json）'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='input.json',
        help='input.json 文件路径（默认: input.json）'
    )
    parser.add_argument(
        '--no-backup', 
        action='store_true',
        help='不备份原 input.json 文件'
    )
    
    args = parser.parse_args()
    
    update_input_from_iter(
        iter_path=args.iter,
        input_path=args.input,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()

