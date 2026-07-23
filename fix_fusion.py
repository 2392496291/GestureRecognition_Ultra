#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动修复CNN和MediaPipe融合的脚本
使用精确的行替换方法
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """备份原文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath + f'.backup_{timestamp}'
    shutil.copy2(filepath, backup_path)
    print(f"✓ 已备份到: {os.path.basename(backup_path)}")
    return backup_path

def fix_fusion():
    """修复融合代码"""
    
    file_path = os.path.join(os.path.dirname(__file__), 'UI_rec', 'SignRecognition.py')
    
    if not os.path.exists(file_path):
        print(f"✗ 错误：找不到文件 {file_path}")
        return False
    
    print("=" * 70)
    print("CNN和MediaPipe融合自动修复工具")
    print("=" * 70)
    print()
    
    # 备份
    backup_path = backup_file(file_path)
    
    try:
        # 读取所有行
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"✓ 文件总行数: {len(lines)}")
        print()
        
        # 定义新代码块（注意保持准确的缩进 - 20个空格）
        new_code_block_1 = """                    # MediaPipe识别结果
                    mp_result = get_str_guester(up_fingers, list_lms)
                    
                    # CNN识别结果
                    cnn_result, cnn_conf = self._get_hand_prediction(frame, bbox)
                    
                    # 融合策略：优先使用高置信度的CNN结果，否则使用MediaPipe
                    final_result, method = self._fuse_predictions(mp_result, cnn_result, cnn_conf)
                    
                    str_guester = final_result
                    self.detInfo.append([str_guester, bbox, method, cnn_conf if cnn_result else 0.0])
"""
        
        new_code_block_2 = """                    # MediaPipe识别结果
                    mp_result = get_str_guester(up_fingers, list_lms)
                    
                    # CNN识别结果
                    cnn_result, cnn_conf = self._get_hand_prediction(image, bbox)
                    
                    # 融合策略：优先使用高置信度的CNN结果，否则使用MediaPipe
                    final_result, method = self._fuse_predictions(mp_result, cnn_result, cnn_conf)
                    
                    str_guester = final_result
                    self.detInfo.append([str_guester, bbox, method, cnn_conf if cnn_result else 0.0])
"""
        
        modifications = 0
        
        # 查找并替换所有匹配的位置
        i = 0
        while i < len(lines):
            # 检查是否匹配目标模式
            if i + 1 < len(lines):
                line1 = lines[i].rstrip()
                line2 = lines[i + 1].rstrip()
                
                if ('str_guester = get_str_guester(up_fingers, list_lms)' in line1 and
                    'self.detInfo.append([str_guester, bbox])' in line2):
                    
                    # 判断是第几处修改（通过前面的代码判断）
                    # 查看前面几行来判断是图片、摄像头还是视频识别
                    context_before = ''.join(lines[max(0, i-50):i])
                    
                    if 'frame = image.copy()' in context_before and modifications == 0:
                        # 第1处：图片识别，使用frame
                        print(f"✓ 找到第1处：图片识别部分 (行 {i+1}-{i+2})")
                        lines[i:i+2] = [new_code_block_1]
                        modifications += 1
                        print(f"  → 使用 frame 参数")
                    else:
                        # 第2、3处：摄像头和视频识别，使用image
                        print(f"✓ 找到第{modifications+1}处：{'摄像头' if modifications == 1 else '视频'}识别部分 (行 {i+1}-{i+2})")
                        lines[i:i+2] = [new_code_block_2]
                        modifications += 1
                        print(f"  → 使用 image 参数")
                    
                    # 跳过已替换的行
                    i += 1
                    continue
            
            i += 1
        
        print()
        print(f"{'='*70}")
        
        if modifications == 3:
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            print(f"✓✓✓ 成功！完成了 {modifications} 处修改")
            print()
            print("修改详情：")
            print("  1. 图片识别部分 - 使用 CNN + MediaPipe 融合")
            print("  2. 摄像头识别部分 - 使用 CNN + MediaPipe 融合")
            print("  3. 视频识别部分 - 使用 CNN + MediaPipe 融合")
            print()
            print(f"原文件已备份到: {os.path.basename(backup_path)}")
            print()
            print("下一步：")
            print("  1. 运行程序: python UI_rec\\runMain.py")
            print("  2. 查看控制台输出确认CNN模型加载")
            print("  3. 测试手势识别功能")
            return True
            
        elif modifications > 0:
            print(f"⚠ 警告：只完成了 {modifications}/3 处修改")
            print("  可能部分代码已经修改过了")
            
            # 仍然写入
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        else:
            print("⚠ 没有找到需要修改的代码")
            print("  可能已经应用过补丁了")
            # 删除备份
            if os.path.exists(backup_path):
                os.remove(backup_path)
            return False
            
    except Exception as e:
        print(f"\n✗✗✗ 错误: {e}")
        print(f"正在恢复备份...")
        # 恢复
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            print("✓ 已恢复原文件")
        return False

if __name__ == "__main__":
    try:
        success = fix_fusion()
        print("=" * 70)
        if success:
            print("\n🎉 修改完成！可以运行程序测试了。\n")
        else:
            print("\n⚠ 修改未完成，请查看上面的提示信息。\n")
    except KeyboardInterrupt:
        print("\n\n已取消操作")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
