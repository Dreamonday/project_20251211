#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
宏观指数数据批量获取脚本 v0.1

功能：
1. 批量获取多个指数的K线数据（日K、周K）
2. 支持设置时间范围（按天数或指定起止日期）
3. 将不同指数的数据存入Excel文件的不同sheet中
4. 自动生成带时间戳的文件名

支持的指数：
- 中国指数：沪深300、上证指数、深证成指、创业板指、中证500、上证50等
- 美国指数：标普500、纳斯达克、道琼斯
- 香港指数：恒生指数、恒生科技

作者：AI助手
创建时间：2025-12-19
版本：v0.1
"""

import os
import sys
import pandas as pd
from datetime import datetime
import time
import importlib.util

# 设置工作目录和路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src", "providers"))

# 导入指数数据获取模块（使用importlib动态导入，避免模块名中的数字问题）
_eastmoney_index_path = os.path.join(BASE_DIR, "src", "providers", "eastmoney_v0.8_index.py")
_eastmoney_index_spec = importlib.util.spec_from_file_location("eastmoney_v0_8_index", _eastmoney_index_path)
eastmoney_index_module = importlib.util.module_from_spec(_eastmoney_index_spec)
_eastmoney_index_spec.loader.exec_module(eastmoney_index_module)
get_index_historical_data = eastmoney_index_module.get_index_historical_data


# ==================== 配置区域（可修改） ====================

# 指数列表配置
# 格式：{'name': '显示名称', 'code': '指数代码', 'period': 'daily'或'weekly'}
INDEX_LIST = [
    # 中国指数
    {'name': '沪深300', 'code': '000300', 'period': 'daily'},
    {'name': '上证指数', 'code': '000001', 'period': 'daily'},
    {'name': '深证成指', 'code': '399001', 'period': 'daily'},
    {'name': '创业板指', 'code': '399006', 'period': 'daily'},
    {'name': '中证500', 'code': '000905', 'period': 'daily'},
    {'name': '上证50', 'code': '000016', 'period': 'daily'},
    
    # 美国指数
    {'name': '标普500', 'code': 'SPX', 'period': 'daily'},
    {'name': '纳斯达克', 'code': 'NDX', 'period': 'daily'},
    {'name': '道琼斯', 'code': 'DJIA', 'period': 'daily'},
    
    # 香港指数
    {'name': '恒生指数', 'code': 'HSI', 'period': 'daily'},
    {'name': '恒生科技', 'code': 'HSTECH', 'period': 'daily'},
]

# 时间范围配置（二选一）
# 方式1：按天数（推荐）
DAYS = 365  # 获取最近N天的数据

# 方式2：指定日期范围
START_DATE = 19000101  # 格式：'YYYYMMDD'，如 '20200101'
END_DATE = 20500101    # 格式：'YYYYMMDD'，如 '20241231'
USE_DATE_RANGE = True  # True=使用日期范围，False=使用天数

# 请求延迟（秒）- 避免请求过快
DELAY_BETWEEN_REQUESTS = 30

# 输出目录（Excel文件保存位置）
OUTPUT_DIR = BASE_DIR  # 直接保存在项目根目录

# ========================================================================


class MacroIndexFetcher:
    """宏观指数数据获取器"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(OUTPUT_DIR, f"macro_index_v0.1_{self.timestamp}.xlsx")
        self.success_data = {}  # {指数名称: DataFrame}
        self.failed_records = []  # [{指数名称, 代码, 错误信息}]
    
    def fetch_single_index(self, index_config):
        """
        获取单个指数的数据
        
        Args:
            index_config (dict): 指数配置，包含name、code、period
        
        Returns:
            tuple: (success: bool, data: DataFrame or None, error_msg: str)
        """
        index_name = index_config['name']
        index_code = index_config['code']
        period = index_config.get('period', 'daily')
        
        try:
            print(f"  📊 获取 {index_name} ({index_code}) 的{period}数据...")
            
            # 根据配置选择时间范围
            if USE_DATE_RANGE and START_DATE and END_DATE:
                df = get_index_historical_data(
                    index_code=index_code,
                    period=period,
                    start_date=START_DATE,
                    end_date=END_DATE
                )
            else:
                df = get_index_historical_data(
                    index_code=index_code,
                    period=period,
                    days=DAYS
                )
            
            if df is not None and not df.empty:
                print(f"    ✅ 成功获取 {len(df)} 条记录")
                return True, df, ""
            else:
                error_msg = "数据为空"
                print(f"    ❌ {error_msg}")
                return False, None, error_msg
                
        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ 获取失败: {error_msg}")
            return False, None, error_msg
    
    def process_all_indices(self):
        """处理所有指数"""
        print("=" * 80)
        print("开始批量获取宏观指数数据")
        print("=" * 80)
        print(f"\n配置信息：")
        print(f"  指数数量: {len(INDEX_LIST)}")
        if USE_DATE_RANGE and START_DATE and END_DATE:
            print(f"  时间范围: {START_DATE} ~ {END_DATE}")
        else:
            print(f"  获取天数: {DAYS} 天")
        print(f"  输出文件: {os.path.basename(self.output_file)}")
        print()
        
        # 遍历所有指数
        for i, index_config in enumerate(INDEX_LIST, 1):
            index_name = index_config['name']
            print(f"[{i}/{len(INDEX_LIST)}] 处理指数: {index_name}")
            
            success, data, error_msg = self.fetch_single_index(index_config)
            
            if success:
                self.success_data[index_name] = data
            else:
                self.failed_records.append({
                    '指数名称': index_name,
                    '指数代码': index_config['code'],
                    '周期': index_config.get('period', 'daily'),
                    '错误信息': error_msg,
                    '处理时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # 延迟，避免请求过快
            if i < len(INDEX_LIST) and DELAY_BETWEEN_REQUESTS > 0:
                time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # 保存到Excel
        self.save_to_excel()
        
        # 输出统计信息
        self.print_summary()
    
    def save_to_excel(self):
        """保存数据到Excel文件"""
        if not self.success_data and not self.failed_records:
            print("\n⚠️  没有数据可保存")
            return
        
        print(f"\n{'='*80}")
        print("保存数据到Excel文件")
        print(f"{'='*80}")
        
        try:
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                # 保存成功的指数数据
                for index_name, df in self.success_data.items():
                    # 清理sheet名称（Excel sheet名称不能包含某些特殊字符）
                    sheet_name = self._clean_sheet_name(index_name)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  ✅ {index_name}: {len(df)} 条记录 -> Sheet: {sheet_name}")
                
                # 保存失败记录
                if self.failed_records:
                    failed_df = pd.DataFrame(self.failed_records)
                    failed_df.to_excel(writer, sheet_name='失败记录', index=False)
                    print(f"  ⚠️  失败记录: {len(self.failed_records)} 条 -> Sheet: 失败记录")
            
            print(f"\n✅ Excel文件已保存: {self.output_file}")
            
        except Exception as e:
            print(f"\n❌ 保存Excel文件失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _clean_sheet_name(self, name):
        """
        清理sheet名称，移除Excel不支持的字符
        
        Excel sheet名称限制：
        - 不能超过31个字符
        - 不能包含: \ / ? * [ ]
        """
        # 移除不支持的字符
        invalid_chars = ['\\', '/', '?', '*', '[', ']', ':']
        cleaned = name
        for char in invalid_chars:
            cleaned = cleaned.replace(char, '_')
        
        # 限制长度
        if len(cleaned) > 31:
            cleaned = cleaned[:31]
        
        return cleaned
    
    def print_summary(self):
        """输出处理结果统计"""
        print(f"\n{'='*80}")
        print("处理结果统计")
        print(f"{'='*80}")
        
        total = len(INDEX_LIST)
        success_count = len(self.success_data)
        failed_count = len(self.failed_records)
        
        print(f"总指数数: {total}")
        print(f"成功: {success_count}")
        print(f"失败: {failed_count}")
        
        if self.success_data:
            print(f"\n✅ 成功获取的指数:")
            for index_name, df in self.success_data.items():
                date_range = f"{df['日期'].iloc[-1]} ~ {df['日期'].iloc[0]}"
                print(f"  - {index_name}: {len(df)} 条记录 ({date_range})")
        
        if self.failed_records:
            print(f"\n❌ 失败的指数:")
            for record in self.failed_records:
                print(f"  - {record['指数名称']} ({record['指数代码']}): {record['错误信息']}")
        
        print(f"\n📁 输出文件: {os.path.basename(self.output_file)}")
        print(f"   完整路径: {self.output_file}")


def main():
    """主函数"""
    try:
        fetcher = MacroIndexFetcher()
        fetcher.process_all_indices()
        
        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  处理被用户中断")
    except Exception as e:
        print(f"\n\n❌ 处理过程中发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
