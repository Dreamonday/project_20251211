"""东方财富数据源模块 v0.8 - 指数K线数据专用版

v0.8 新增功能：
1. ✅ 支持获取指数K线数据（日K、周K）
2. ✅ 支持中国指数（沪深300、上证指数、深证成指、创业板指、中证500、上证50等）
3. ✅ 支持美国指数（标普500、纳斯达克、道琼斯）
4. ✅ 支持香港指数（恒生指数、恒生科技）
5. ✅ 智能指数代码识别和secid格式转换
6. ✅ 复用v0.7的错误处理和请求逻辑

基于测试脚本验证的secid格式：
- 中国指数：使用市场代码0（深圳）、1（上海）、90（部分指数）
- 美国指数：使用市场代码100、106、107
- 香港指数：使用市场代码100、124

提供指数K线数据获取功能。
"""

import random
import requests
import pandas as pd
from datetime import datetime, timedelta
import json


# ==================== 全局配置 ====================

# 全局Session对象（连接复用）
_session = None

# Session模式开关：False=每次新建连接（更隐蔽，但速度慢），True=复用连接（快但可能被识别）
USE_SESSION = False


# ==================== 指数代码映射表 ====================

# 基于测试结果构建的指数代码映射表
# 格式：{指数名称或代码: (市场代码, 指数代码)}
INDEX_MAPPING = {
    # 中国指数
    '沪深300': ('1', '000300'),
    'CSI300': ('1', '000300'),
    '000300': ('1', '000300'),
    
    '上证指数': ('1', '000001'),  # 也支持0.000001，但1.000001更常用
    'SH000001': ('1', '000001'),
    '000001': ('1', '000001'),
    
    '深证成指': ('0', '399001'),
    'SZ399001': ('0', '399001'),
    '399001': ('0', '399001'),
    
    '创业板指': ('0', '399006'),
    'SZ399006': ('0', '399006'),
    '399006': ('0', '399006'),
    
    '中证500': ('1', '000905'),  # 也支持0.000905和90.000905，但1.000905更常用
    'CSI500': ('1', '000905'),
    '000905': ('1', '000905'),
    
    '上证50': ('1', '000016'),  # 也支持0.000016，但1.000016更常用
    'SSE50': ('1', '000016'),
    '000016': ('1', '000016'),
    
    # 美国指数
    '标普500': ('100', 'SPX'),
    'SPX': ('100', 'SPX'),
    'S&P500': ('100', 'SPX'),
    'SP500': ('100', 'SPX'),
    
    '纳斯达克': ('100', 'NDX'),
    'NASDAQ': ('100', 'NDX'),
    'NDX': ('100', 'NDX'),
    'IXIC': ('100', 'NDX'),
    
    '道琼斯': ('100', 'DJIA'),  # 也支持107.DJIA，但100.DJIA更常用
    'DJI': ('100', 'DJIA'),
    'DJIA': ('100', 'DJIA'),
    'DOW': ('100', 'DJIA'),
    
    # 香港指数
    '恒生指数': ('100', 'HSI'),
    'HSI': ('100', 'HSI'),
    'HANG SENG': ('100', 'HSI'),
    
    '恒生科技': ('124', 'HSTECH'),
    'HSTECH': ('124', 'HSTECH'),
    '恒生科技指数': ('124', 'HSTECH'),
}


def _get_session():
    """获取或创建全局Session对象"""
    global _session
    if _session is None:
        _session = requests.Session()
        # 设置连接池大小
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=0  # 不自动重试，由上层控制
        )
        _session.mount('http://', adapter)
        _session.mount('https://', adapter)
    return _session


# ==================== 动态请求头生成 ====================

def _get_random_headers():
    """
    生成随机化的HTTP请求头
    
    Returns:
        dict: HTTP请求头字典
    """
    # User-Agent池：30个常见浏览器（Chrome、Firefox、Safari、Edge）
    user_agents = [
        # Chrome - Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        
        # Chrome - Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        
        # Chrome - Linux
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        
        # Firefox - Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0',
        
        # Firefox - Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 13.6; rv:121.0) Gecko/20100101 Firefox/121.0',
        
        # Safari - Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15',
        
        # Edge - Windows
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
        
        # Edge - Mac
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
        
        # 其他版本混合
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    ]
    
    # Referer池：东方财富常见页面
    referers = [
        'https://quote.eastmoney.com/',
        'https://data.eastmoney.com/',
        'https://www.eastmoney.com/',
        'https://quote.eastmoney.com/center/',
        'https://data.eastmoney.com/stockdata/',
    ]
    
    headers = {
        'User-Agent': random.choice(user_agents),
        'Referer': random.choice(referers),
    }
    
    return headers


# ==================== 指数代码识别和转换 ====================

def normalize_index_code(index_code):
    """
    标准化指数代码，转换为大写并去除空格
    
    Args:
        index_code (str): 指数代码或名称
    
    Returns:
        str: 标准化后的指数代码
    """
    if not index_code:
        return None
    return str(index_code).strip().upper()


def get_index_secid(index_code):
    """
    根据指数代码或名称获取secid格式
    
    Args:
        index_code (str): 指数代码或名称，如 '沪深300', '000300', 'SPX', 'HSI'
    
    Returns:
        tuple: (secid, market_code, index_code) 或 None（如果未找到）
            - secid: 完整的secid格式，如 '1.000300'
            - market_code: 市场代码，如 '1'
            - index_code: 指数代码，如 '000300'
    """
    normalized = normalize_index_code(index_code)
    if not normalized:
        return None
    
    # 首先尝试直接匹配
    if normalized in INDEX_MAPPING:
        market_code, code = INDEX_MAPPING[normalized]
        secid = f'{market_code}.{code}'
        return secid, market_code, code
    
    # 如果未找到，尝试智能识别中国指数（6位数字代码）
    if normalized.isdigit() and len(normalized) == 6:
        # 中国指数代码规则
        prefix = normalized[:3]
        if prefix == '000':
            # 000开头的指数，优先使用市场代码1（上海）
            return f'1.{normalized}', '1', normalized
        elif prefix == '399':
            # 399开头的指数，使用市场代码0（深圳）
            return f'0.{normalized}', '0', normalized
    
    return None


# ==================== 指数K线数据获取 ====================

def get_index_historical_data(index_code, period='daily', start_date=None, end_date=None, days=365):
    """
    获取指数历史K线数据
    
    v0.8 新增功能：
    - 支持中国指数、美国指数、香港指数
    - 自动识别指数代码并转换为正确的secid格式
    - 复用v0.7的错误处理和请求逻辑
    
    Args:
        index_code (str): 指数代码或名称
            - 中国指数：'沪深300', '000300', 'CSI300', '上证指数', '000001', '深证成指', '399001' 等
            - 美国指数：'标普500', 'SPX', 'S&P500', '纳斯达克', 'NASDAQ', 'COMP', '道琼斯', 'DJI', 'DJIA' 等
            - 香港指数：'恒生指数', 'HSI', '恒生科技', 'HSTECH' 等
        period (str): 数据周期。可选值：
            - 'daily': 日K线（默认）
            - 'weekly': 周K线
        start_date (str): 开始日期，格式 'YYYYMMDD'。如果为None，则使用days参数计算
        end_date (str): 结束日期，格式 'YYYYMMDD'。如果为None，则使用当前日期
        days (int): 获取天数，默认365天。仅在start_date为None时使用
    
    Returns:
        pd.DataFrame: 历史K线数据，包含以下列：
            - 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额
            - 振幅, 涨跌幅, 涨跌额, 换手率
    
    示例：
        >>> df = get_index_historical_data('沪深300', period='daily', days=365)
        >>> df = get_index_historical_data('SPX', period='weekly', start_date='20200101', end_date='20231231')
        >>> df = get_index_historical_data('000300', period='daily', days=180)
        >>> df = get_index_historical_data('HSI', period='daily', days=730)
    """
    # 获取secid格式
    secid_info = get_index_secid(index_code)
    if secid_info is None:
        raise ValueError(f"不支持的指数代码或名称: {index_code}")
    
    secid, market_code, code = secid_info
    
    # 计算日期范围
    if end_date is None:
        end_date_dt = datetime.now()
    else:
        try:
            end_date_dt = datetime.strptime(str(end_date), "%Y%m%d")
        except ValueError:
            raise ValueError(f"无效的结束日期格式: {end_date}，应为YYYYMMDD")
    
    if start_date is None:
        start_date_dt = end_date_dt - timedelta(days=days)
    else:
        try:
            start_date_dt = datetime.strptime(str(start_date), "%Y%m%d")
        except ValueError:
            raise ValueError(f"无效的开始日期格式: {start_date}，应为YYYYMMDD")
    
    start_date_str = start_date_dt.strftime("%Y%m%d")
    end_date_str = end_date_dt.strftime("%Y%m%d")
    
    # 调用内部函数获取数据
    return _get_index_kline_data(secid, period, start_date_str, end_date_str, index_code)


def _get_index_kline_data(secid, period, start_date, end_date, index_code=None):
    """
    获取指数K线数据（内部函数）
    
    Args:
        secid (str): 证券ID，格式如 '1.000300'
        period (str): 周期类型，'daily'=日K, 'weekly'=周K
        start_date (str): 开始日期，格式 'YYYYMMDD'
        end_date (str): 结束日期，格式 'YYYYMMDD'
        index_code (str): 指数代码，用于错误信息显示
    
    Returns:
        pd.DataFrame: K线数据
    """
    # 周期映射
    klt_map = {
        'daily': 101,   # 日K
        'weekly': 102,  # 周K
    }
    klt = klt_map.get(period, 101)
    
    # 构建API URL
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    
    # 请求参数
    params = {
        'secid': secid,
        'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': klt,
        'fqt': 0,  # 0=不复权
        'beg': start_date,
        'end': end_date,
        '_': str(int(datetime.now().timestamp() * 1000)),
    }
    
    try:
        # 使用动态请求头
        headers = _get_random_headers()
        
        # 根据配置选择使用 Session 或直接请求
        if USE_SESSION:
            session = _get_session()
            response = session.get(url, params=params, headers=headers, timeout=30)
        else:
            response = requests.get(url, params=params, headers=headers, timeout=30)
        
        # 检查HTTP状态码
        if response.status_code != 200:
            error_msg = f"HTTP错误 {response.status_code}"
            if index_code:
                error_msg += f" (指数: {index_code}, secid: {secid})"
            raise requests.HTTPError(error_msg)
        
        # 检查返回内容
        if not response.text:
            raise ValueError("API返回空内容")
        
        # 尝试解析JSON
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"响应内容不是有效的JSON格式: {e}")
        
        # 解析数据
        if 'data' not in data or not data['data']:
            raise ValueError("响应JSON中缺少data字段")
        
        klines = data['data'].get('klines', [])
        if not klines:
            raise ValueError(f"返回的K线数据为空 (指数: {index_code or secid}, 日期范围: {start_date}→{end_date})")
        
        # 解析K线数据
        records = []
        parse_errors = 0
        for kline in klines:
            parts = kline.split(',')
            if len(parts) >= 11:
                try:
                    records.append({
                        '日期': parts[0],
                        '开盘': float(parts[1]),
                        '收盘': float(parts[2]),
                        '最高': float(parts[3]),
                        '最低': float(parts[4]),
                        '成交量': float(parts[5]),
                        '成交额': float(parts[6]),
                        '振幅': float(parts[7]),
                        '涨跌幅': float(parts[8]),
                        '涨跌额': float(parts[9]),
                        '换手率': float(parts[10]),
                    })
                except (ValueError, IndexError):
                    parse_errors += 1
            else:
                parse_errors += 1
        
        if parse_errors > 0:
            print(f"⚠️  解析警告: {parse_errors}/{len(klines)} 条K线数据格式异常，已跳过")
        
        if not records:
            raise ValueError("所有K线数据解析失败")
        
        return pd.DataFrame(records)
        
    except requests.RequestException as req_err:
        error_msg = f"网络请求失败: {req_err}"
        if index_code:
            error_msg += f" (指数: {index_code}, secid: {secid})"
        raise requests.RequestException(error_msg) from req_err
    except Exception as exc:
        error_msg = str(exc)
        if index_code:
            error_msg += f" (指数: {index_code}, secid: {secid})"
        raise type(exc)(error_msg) from exc


# ==================== 辅助函数 ====================

def list_supported_indices():
    """
    列出所有支持的指数
    
    Returns:
        dict: 支持的指数字典，格式为 {指数名称: (市场代码, 指数代码)}
    """
    return INDEX_MAPPING.copy()


def is_index_supported(index_code):
    """
    检查指数代码是否支持
    
    Args:
        index_code (str): 指数代码或名称
    
    Returns:
        bool: 如果支持返回True，否则返回False
    """
    return get_index_secid(index_code) is not None
