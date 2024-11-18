print('开始导入必要模块...')
import torch
import random
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sqlite3
from datetime import datetime
import itertools
import time
import pickle
print('模块导入完成')

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class DataProcessor:
    def __init__(self, feature_dir: str, target_path: str):
        self.feature_dir = feature_dir
        self.target_path = target_path
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # 为y单独创建一个scaler
        
    def load_features(self) -> pd.DataFrame:
        """加载所有特征数据"""
        dir_list = os.listdir(self.feature_dir)
        df_lists = []
        for name in dir_list:
            csv_path = os.path.join(self.feature_dir, name)
            csv_df = pd.read_csv(csv_path, usecols=['Close'])
            df_lists.append(csv_df)
        
        df = pd.concat(df_lists, axis=1, ignore_index=True)
        df.columns = [name[:6] for name in dir_list]
        return df
    
    def load_target(self) -> pd.DataFrame:
        """加载目标变量"""
        return pd.read_csv(self.target_path, usecols=['Close'])
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值
        1. 先用前向填充处理短期缺失
        2. 对较长的缺失段使用线性插值
        3. 处理开头和结尾的缺失值
        """
        df = df.ffill(limit=3)
        df = df.interpolate(method='linear', limit_direction='both', limit=5)
        return df.ffill().bfill()
    
    def prepare_data(self) -> pd.DataFrame:
        """完整的数据准备流程"""
        # 加载数据
        features_df = self.load_features()
        target_df = self.load_target()
        
        # 验证数据长度是否匹配
        if len(features_df) != len(target_df):
            raise ValueError("Features and target data have different lengths")
        
        # 合并特征和目标
        df = features_df.copy()
        df['000300'] = target_df['Close'].values
        
        # 处理缺失值
        df = self.handle_missing_values(df)
        
        # 验证是否还有缺失值
        if df.isnull().any().any():
            raise ValueError("Data still contains missing values after processing")
        
        return df
    
    def split_and_scale(self, df: pd.DataFrame, test_size: float = 0.2):
        """按时间顺序划分数据集并进行标准化"""
        X = df.iloc[:, :-1]  # 特征
        y = df.iloc[:, -1]   # 目标变量
        
        # 计算划分点
        split_idx = int(len(df) * (1 - test_size))
        
        # 按时间顺序划分数据集
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_val = y.iloc[split_idx:]
        
        # 使用训练集的统计数据进行标准化
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        # 返回DataFrame格式的数据
        return (
            pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index),
            pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index),
            y_train.values,
            y_val.values
        )
    
    def inverse_transform_y(self, y_scaled):
        """将标准化的y转换回原始尺度"""
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

class LassoModel:
    def __init__(self, n_features: int, learning_rate: float, l1_lambda: float, epochs: int, step_size: int = 500, gamma: float = 0.9):
        # 修改权重初始化方式，使用更合适的初始化范围
        self.weights = torch.nn.Parameter(torch.randn(n_features + 1, device=device) * 0.01)
        
        # 使用Adam优化器，添加momentum
        self.optimizer = torch.optim.Adam(
            [self.weights], 
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用StepLR调度器，每step_size个epoch将学习率乘以gamma
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
        self.l1_lambda = l1_lambda

    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        # 添加一列1作为偏置项，确保在GPU上
        ones = torch.ones(X_batch.shape[0], 1, device=device)
        X_with_bias = torch.cat([X_batch, ones], dim=1)
        return X_with_bias @ self.weights

    def compute_loss(self, output: torch.Tensor, y_batch: torch.Tensor) -> tuple:
        """使MSE损失和L1正则化"""
        # 计算相对误差而不是绝对误
        relative_error = ((output - y_batch) / y_batch).pow(2).mean()
        
        # L1正则化
        l1_loss = self.l1_lambda * torch.sum(torch.abs(self.weights[:-1]))
        
        return relative_error + l1_loss, relative_error

    def get_top_stocks(self, stock_path, top_n=30):
        """获取权重最大的前N只股票信息"""
        try:
            # 读取股票信息文件
            stock_info = pd.read_csv(stock_path, encoding='gbk')
            
            # 获取权重最大的前N个特征的索引
            weights_cpu = self.weights.cpu().detach().numpy()
            
            # 打印调试信息
            print(f"\nWeights shape: {weights_cpu.shape}")
            print(f"Feature names length: {len(self.feature_names)}")
            
            # 确保不超过可用的特征数量
            actual_top_n = min(top_n, len(self.feature_names))
            # 只选择有效范围内的索引
            valid_indices = np.where(np.abs(weights_cpu) > 0)[0]
            valid_indices = valid_indices[valid_indices < len(self.feature_names)]
            
            # 按权重绝对值排序并取前N个
            top_indices = sorted(valid_indices, key=lambda x: abs(weights_cpu[x]), reverse=True)[:actual_top_n]
            
            # 获取这些特征对应的股票代码和权重
            top_stocks_data = []
            for idx in top_indices:
                stock_code = str(self.feature_names[idx]).zfill(6)
                weight = weights_cpu[idx]
                
                # 添加到结果列表
                top_stocks_data.append({
                    'Stock_Name': stock_code,
                    'Weight': abs(weight)
                })
            
            # 创建DataFrame并按权重排序
            top_stocks_df = pd.DataFrame(top_stocks_data)
            if not top_stocks_df.empty:
                # 确保股票代码格式正确
                top_stocks_df['Stock_Name'] = top_stocks_df['Stock_Name'].astype(str).str.zfill(6)
                
                # 合并股票信息
                stock_info['code'] = stock_info['code'].astype(str).str.zfill(6)
                merged_df = pd.merge(
                    top_stocks_df,
                    stock_info[['code', 'name']],
                    left_on='Stock_Name',
                    right_on='code',
                    how='left'
                )
                
                # 整理最终结果
                merged_df = merged_df.rename(columns={'name': '股票中文名'})
                merged_df = merged_df.drop(columns=['code'])
                merged_df = merged_df.sort_values('Weight', ascending=False)
                
                return merged_df
                
            return pd.DataFrame(columns=['Stock_Name', '股票中文名', 'Weight'])
            
        except Exception as e:
            print(f"\nError in get_top_stocks with details: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['Stock_Name', '股票中文名', 'Weight'])

class ModelTrainer:
    def __init__(self, model: LassoModel = None, patience: int = 10):
        self.model = model
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_weights = None
        self.best_epoch = 0
        self.data_processor = None
        self.feature_names = None  
        
        # 将模型移到GPU
        if self.model is not None:
            self.model.weights = self.model.weights.to(device)

    def train_epoch(self, train_loader, val_loader, epoch, data_processor):
        """训练一个epoch并返回训练和验证损失"""
        # 训练阶段
        train_total_loss = 0
        train_mse_loss = 0
        
        for X_batch, y_batch in train_loader:
            self.model.optimizer.zero_grad()
            output = self.model.forward(X_batch)
            loss, mse = self.model.compute_loss(output, y_batch)
            loss.backward()
            self.model.optimizer.step()
            train_total_loss += loss.item()
            train_mse_loss += mse.item()
        
        # 更新学习率
        self.model.scheduler.step()
        
        # 验证阶段
        val_total_loss = 0
        val_mse_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = self.model.forward(X_batch)
                loss, mse = self.model.compute_loss(output, y_batch)
                val_total_loss += loss.item()
                val_mse_loss += mse.item()
        
        # 计算平均损失
        train_loss = train_total_loss / len(train_loader)
        val_loss = val_total_loss / len(val_loader)
        
        # 获取当前学习率
        current_lr = self.model.optimizer.param_groups[0]['lr']
        
        # 每100个epoch计算和打印详细信息
        if epoch % 100 == 0:
            # 收集所有验证集数据用于计算跟踪误差
            X_val_full = torch.cat([batch[0] for batch in val_loader], dim=0)
            y_val_full = torch.cat([batch[1] for batch in val_loader], dim=0)
            
            # 计算跟踪误差
            tracking_error = calculate_tracking_error(self.model, X_val_full, y_val_full, data_processor)
            
            # 打印训练信息
            print(f'Epoch: {epoch:6d} | '
                  f'Train Loss: {train_loss:.8f} | '
                  f'Val Loss: {val_loss:.8f} | '
                  f'LR: {current_lr:.8f} | '
                  f'TE: {tracking_error:.2f} bp | '
                  f'Best Val Loss: {self.best_val_loss:.8f} | '
                  f'Patience: {self.patience_counter}/{self.patience}')
            
            # 可选：保存到日志文件
            with open('training_log.txt', 'a') as f:
                f.write(f'{epoch},{train_loss:.8f},{val_loss:.8f},{current_lr:.8f},{tracking_error:.2f}\n')
        
        return train_loss, val_loss

    def get_feature_importance(self, feature_names, scaler):
        """计算特征重要性"""
        with torch.no_grad():
            # 获型权重
            weights = self.model.weights[:-1].numpy()  # 不包括偏置项
            
            # 如果使用了标准化，需要调整权重
            if scaler:
                weights = weights * scaler.scale_
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(weights)
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=1000, data_processor=None):
        """完整训练流程"""
        # 保存特征名称
        self.feature_names = X_train.columns.tolist()
        self.model.feature_names = self.feature_names  # 设置模型的feature_names
        
        # 确保数据是正确的格式
        X_train_data = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_data = y_train.values if hasattr(y_train, 'values') else y_train
        X_val_data = X_val.values if hasattr(X_val, 'values') else X_val
        y_val_data = y_val.values if hasattr(y_val, 'values') else y_val
        
        # 1. 创建训练集的数据集对象，并移到GPU
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_data).to(device),
            torch.FloatTensor(y_train_data).to(device)
        )
        
        # 2. 创建训练集的数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )
        
        # 3. 创建验证集的数据集对象，并移到GPU
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val_data).to(device),
            torch.FloatTensor(y_val_data).to(device)
        )
        
        # 4. 创建验证集的数据加载器
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        train_losses = []
        val_losses = []
        tracking_errors = []
        learning_rates = []
        
        # 保存data_processor作为类属性
        self.data_processor = data_processor
        
        print(f"Starting training with {epochs} epochs...")
        for epoch in range(epochs):
            if epoch % 100 == 0:  # 每100个epoch显示一次进度
                print(f"Progress: {epoch}/{epochs} epochs ({epoch/epochs*100:.1f}%)")
            
            train_loss, val_loss = self.train_epoch(
                train_loader, 
                val_loader, 
                epoch, 
                self.data_processor  # 使用保存的data_processor
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 记录学习率
            current_lr = self.model.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # 每100个epoch计算跟踪误差
            if epoch % 100 == 0:
                te = calculate_tracking_error(self.model, X_val, y_val, data_processor)
                tracking_errors.append(te)
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = self.model.weights.detach().clone()
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # 恢复最佳权重
        self.model.weights.data = self.best_weights
        
        # 绘制训练过程图
        self.plot_training_process(train_losses, val_losses, tracking_errors, learning_rates)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'tracking_errors': tracking_errors,
            'learning_rates': learning_rates,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_weights': self.best_weights
        }

    def plot_training_process(self, train_losses, val_losses, tracking_errors, learning_rates):
        """绘制训练过程的各项指标"""
        # 创建保存图片的目录
        image_dir = 'results/images'
        os.makedirs(image_dir, exist_ok=True)
        
        # 构建文件名
        params_str = f"lr{self.model.optimizer.param_groups[0]['lr']}_" \
                     f"l1_{self.model.l1_lambda}_" \
                     f"batch_{len(train_losses)}_" \
                     f"patience_{self.patience}_" \
                     f"{time.strftime('%Y%m%d_%H%M%S')}"
        
        save_path = os.path.join(image_dir, f"training_process_{params_str}.png")
        
        # 绘图逻辑
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 绘制训练损失和验证损失
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制学习率变化
        ax2.plot(learning_rates)
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
        
        # 跟踪误差
        epochs = list(range(0, len(train_losses), 100))
        ax3.plot(epochs, tracking_errors)
        ax3.set_title('Tracking Error')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Tracking Error (bp)')
        ax3.grid(True)
        
        # 绘制权重分布
        weights = self.best_weights.cpu().detach().numpy()[:-1]  # 不包括偏置项
        ax4.hist(weights, bins=30)
        ax4.set_title('Feature Weights Distribution')
        ax4.set_xlabel('Weight Value')
        ax4.set_ylabel('Count')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Training process plot saved to: {save_path}")

    def save_model(self, path):
        """保存模型权重和标准化器"""
        model_state = {
            'weights': self.model.weights.data,
            'feature_scaler': self.data_processor.feature_scaler,
            'target_scaler': self.data_processor.target_scaler
        }
        torch.save(model_state, path)

    def load_model(self, path):
        """加载模型权重和标准化器"""
        model_state = torch.load(path)
        self.model.weights.data = model_state['weights']
        self.data_processor.feature_scaler = model_state['feature_scaler']
        self.data_processor.target_scaler = model_state['target_scaler']

def fix_chinese_encoding(text):
    """
    修复中文编码问题
    Args:
        text: 输入文本
    Returns:
        修复后的文本
    """
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        try:
            return text.encode('latin1').decode('gbk')
        except:
            return text

def plot_losses(train_losses, val_losses, best_epoch, save_path=None):
    """
    绘制损失下曲线
    观察模训练过程否正常
    判断模型是否拟（训练损失持续下降而验证损失上升）
    确认早停机制是否在合适的时间点触发
    评估学率等超参数的选择否合适
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_tracking_error(model, X_val, y_val, data_processor):
    """计算跟踪误差"""
    with torch.no_grad():
        try:
            # 确保输入数据是正确的格式并在GPU上
            if isinstance(X_val, torch.Tensor):
                X_val_tensor = X_val.to(device)
            else:
                X_val_tensor = torch.FloatTensor(
                    X_val.values if hasattr(X_val, 'values') else X_val
                ).to(device)
            
            # 获取预测值
            y_pred = model.forward(X_val_tensor)
            
            # 将预测值移回CPU并转换为numpy数组
            y_pred = y_pred.cpu().numpy()
            
            # 如果y_val是torch.Tensor，转���为numpy
            if isinstance(y_val, torch.Tensor):
                y_true = y_val.cpu().numpy()
            else:
                y_true = y_val
            
            # 计算跟踪误差（年化）
            tracking_error = np.sqrt(252) * np.std((y_pred - y_true) / y_true) * 10000
            
            return tracking_error
            
        except Exception as e:
            print(f"Error in calculate_tracking_error: {str(e)}")
            return float('inf')

def save_results_to_db(params, results, db_path='results.db'):
    """保存结果到SQLite数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建（如果不存在）
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS training_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        params TEXT,
        best_val_loss REAL,
        best_epoch INTEGER,
        tracking_error REAL,
        train_time REAL,
        final_learning_rate REAL,
        model_weights BLOB,
        train_losses BLOB,
        val_losses BLOB
    )
    ''')
    
    # 将numpy组转换为二进制
    weights_binary = pickle.dumps(results['best_weights'].cpu().numpy())
    train_losses_binary = pickle.dumps(results['train_losses'])
    val_losses_binary = pickle.dumps(results['val_losses'])
    
    # 检查是否存在相同数的记录
    cursor.execute(
        "SELECT best_val_loss FROM training_results WHERE params = ?", 
        (str(params),)
    )
    existing_record = cursor.fetchone()
    
    # 如果不存在相同参数或新结果更好，则保存
    if not existing_record or results['best_val_loss'] < existing_record[0]:
        cursor.execute('''
        INSERT INTO training_results 
        (timestamp, params, best_val_loss, best_epoch, tracking_error, train_time,
         final_learning_rate, model_weights, train_losses, val_losses)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            str(params),
            results['best_val_loss'],
            results['best_epoch'],
            results['tracking_error'],
            results['train_time'],
            results['learning_rates'][-1],
            weights_binary,
            train_losses_binary,
            val_losses_binary
        ))
        
    conn.commit()
    conn.close()

def generate_param_combinations(param_grid):  # Add param_grid as parameter
    """生成参数组合"""
    # 生成所有可能的参数合
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    return combinations

def save_training_results(results_list, save_path, top_n):
    """保存所有训练结果到CSV文件"""
    # 按跟踪误差排序
    results_list.sort(key=lambda x: x['tracking_error'])
    
    # 创建基础列
    df_data = {
        '序号': range(1, len(results_list) + 1),
        'epochs': [r['params']['epochs'] for r in results_list],
        'batch_size': [r['params']['batch_size'] for r in results_list],
        'learning_rate': [r['params']['learning_rate'] for r in results_list],
        'l1_lambda': [r['params']['l1_lambda'] for r in results_list],
        'patience': [r['params']['patience'] for r in results_list],
        'TE': [r['tracking_error'] for r in results_list]
    }
    
    # 先添加所有股票名称
    for i in range(top_n):
        df_data[f'股票名称_{i+1}'] = []
        for r in results_list:
            if 'top_stocks' in r and not r['top_stocks'].empty and i < len(r['top_stocks']):
                df_data[f'股票名称_{i+1}'].append(r['top_stocks']['股票中文名'].iloc[i] 
                                              if '股票中文名' in r['top_stocks'].columns else 'N/A')
            else:
                df_data[f'股票名称_{i+1}'].append('N/A')
    
    # 再添加所有股票代码
    for i in range(top_n):
        df_data[f'股票代码_{i+1}'] = []
        for r in results_list:
            if 'top_stocks' in r and not r['top_stocks'].empty and i < len(r['top_stocks']):
                df_data[f'股票代码_{i+1}'].append(r['top_stocks']['Stock_Name'].iloc[i] 
                                              if 'Stock_Name' in r['top_stocks'].columns else 'N/A')
            else:
                df_data[f'股票代码_{i+1}'].append('N/A')
    
    # 最后添加权重
    for i in range(top_n):
        df_data[f'权重_{i+1}'] = []
        for r in results_list:
            if 'top_stocks' in r and not r['top_stocks'].empty and i < len(r['top_stocks']):
                df_data[f'权重_{i+1}'].append(r['top_stocks']['Weight'].iloc[i] 
                                          if 'Weight' in r['top_stocks'].columns else 0.0)
            else:
                df_data[f'权重_{i+1}'].append(0.0)
    
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建DataFrame并保存
    results_df = pd.DataFrame(df_data)
    results_df.to_csv(save_path, index=False, encoding='UTF-8')
    
    return results_df

def main(**params):
    try:
        # 设置全局随机种子
        seed = params['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        start_time = time.time()  # 添加计时

        # 1. 数据准备和数据预处理
        data_processor = DataProcessor(params['feature_dir'], params['target_path'])
        
        # 2. 生成df对象分割数据集
        df = data_processor.prepare_data()
        X_train, X_val, y_train, y_val = data_processor.split_and_scale(df, test_size=0.2)

        # 3. 初始化模型和训练器
        model = LassoModel(
            n_features=X_train.shape[1],
            learning_rate=params['learning_rate'],
            l1_lambda=params['l1_lambda'],
            epochs=params['epochs'],
            step_size=params['step_size'],
            gamma=params['gamma']
        )
        
        trainer = ModelTrainer(model, patience=params['patience'])
        
        # 4. 训练模型
        training_results = trainer.train(
            X_train, 
            y_train, 
            X_val, 
            y_val, 
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            data_processor=data_processor
        )
        
        # 6. 计算跟踪误差
        tracking_error = calculate_tracking_error(model, X_val, y_val, data_processor)
        
        # 获取top股票，确保传入正确的参数
        try:
            print("\nGetting top stocks with parameters:")
            print(f"stock_info_path: {params['stock_info_path']}")
            print(f"top_n_stocks: {params['top_n_stocks']}")
            
            top_stocks = trainer.model.get_top_stocks(
                stock_path=params['stock_info_path'],
                top_n=params['top_n_stocks']
            )
            print("\nTop stocks DataFrame shape:", top_stocks.shape)
            print("Top stocks columns:", top_stocks.columns.tolist())
            print("\nFirst few rows of top_stocks:")
            print(top_stocks.head())
        except Exception as e:
            print(f"Error in get_top_stocks: {str(e)}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
            top_stocks = pd.DataFrame(columns=['Stock_Name', '股票中文名', 'Weight'])
        
        # 7. 整理结果
        results = {
            'best_val_loss': training_results['best_val_loss'],
            'best_epoch': training_results['best_epoch'],
            'tracking_error': tracking_error,
            'train_time': time.time() - start_time,
            'best_weights': training_results['best_weights'],
            'train_losses': training_results['train_losses'],
            'val_losses': training_results['val_losses'],
            'learning_rates': training_results['learning_rates'],
            'top_stocks': top_stocks,
            'params': params
        }
        
        return results

    except Exception as e:
        print(f"Error in main function: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        
        # 返回一个带有默认值的结果字典
        return {
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'tracking_error': float('inf'),
            'train_time': 0,
            'best_weights': None,
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'top_stocks': pd.DataFrame(columns=['Stock_Name', '股票中文名', 'Weight']),
            'params': params
        }

if __name__ == "__main__":
    # 基础参数配置
    base_params = {
        'feature_dir': r'D:/something/task/task2_lasso_regression/data_utf8',
        'target_path': r'D:/something/task/task2_lasso_regression/data_033000_utf8/sh000300.csv',
        'stock_info_path': r'D:\something\task\task2_lasso_regression\data_000300\000300const_20241115.csv',
        'results_save_path': r'D:/something/task/task2_lasso_regression/results/training_results.csv',
        'top_n_stocks': 30,
        'epochs': 30000,
        'seed': 42
    }
    # 定义参数搜索网格
    # param_grid = {
    #     'learning_rate': [0.001, 0.01, 0.1],
    #     'l1_lambda': [0.01, 0.1, 1.0],
    #     'batch_size': [32, 64, 128],
    #     'step_size': [300, 500, 700],
    #     'gamma': [0.8, 0.9, 0.95],
    #     'patience': [30, 50, 70]
    # }
    # param_grid = {
    #     'learning_rate': [0.001,0.01,0.1],  # 降低学习率
    #     'l1_lambda': [0.1, 0.5, 1.0],     # 降低L1正则化强度
    #     'batch_size': [64, 128, 256],       # 减小batch size
    #     'patience': [100, 300,600,1000,5000,20000],       # 减少patience
    #     'step_size': [500, 400, 300],       # 更频繁地调整学习率
    #     'gamma': [0.9, 0.95]           # 更温和的学习率衰减
    # }
    param_grid = {
        'learning_rate': [0.001],  # 降低学习率
        'l1_lambda': [0.1],     # 降低L1正则化强度
        'batch_size': [64],       # 减小batch size
        'patience': [100],       # 减少patience
        'step_size': [500],       # 更频繁地调整学习率
        'gamma': [0.9]           # 更温和的学习率衰减
    }

    # 获取所有参数组合
    param_combinations = generate_param_combinations(param_grid)
    print(f"Total combinations to test: {len(param_combinations)}")

    # 验证所有必需参数是否存在
    required_params = [
        'learning_rate', 'l1_lambda', 'batch_size', 'patience',
        'epochs', 'seed', 'feature_dir', 'target_path',
        'step_size', 'gamma', 'stock_info_path', 'results_save_path',
        'top_n_stocks'
    ]
    
    # 存储所有训练结果
    all_results = []
    
    # 遍历所有参数组合
    for param_set in param_combinations:
        current_params = {**base_params, **param_set}
        
        # 验证参数
        missing_params = [param for param in required_params if param not in current_params]
        if missing_params:
            print(f"Missing required parameters: {missing_params}")
            continue
            
        print(f"\nTesting parameters: {param_set}")
        
        try:
            results = main(**current_params)
            all_results.append(results)
            print(f"Training completed successfully.")
            print(f"Best validation loss: {results['best_val_loss']:.6f}")
            print(f"Tracking error: {results['tracking_error']:.2f} bp")
            print(f"Training time: {results['train_time']:.2f} seconds")
        except Exception as e:
            print(f"Error occurred with parameters {param_set}: {str(e)}")
            continue
    
    # 保存所有结果到CSV
    results_df = save_training_results(
        all_results,
        save_path=base_params['results_save_path'],
        top_n=base_params['top_n_stocks']
    )
    print(f"\nResults saved to {base_params['results_save_path']}")
