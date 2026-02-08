import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class MarketPlotter:
    """
    Analysis Plotter (Refactored for Final Integration)

    核心功能：
    1. 读取 evaluation_metrics.csv 生成全景实验报告。
    2. 绘制利润构成分析 (Profit Composition)。
    3. 绘制成本结构分析 (Cost Structure)。
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "evaluation_metrics.csv")

        # 绘图风格设置
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'figure.figsize': (12, 8), 'font.size': 12})

    def load_data(self) -> Optional[pd.DataFrame]:
        if not os.path.exists(self.csv_path):
            print(f"Error: Metrics file not found at {self.csv_path}")
            return None
        return pd.read_csv(self.csv_path)

    def plot_all(self):
        """一键生成所有图表"""
        df = self.load_data()
        if df is None:
            return

        print("Generating Analysis Plots...")
        self.plot_profit_composition(df)
        self.plot_cost_breakdown(df)
        self.plot_soc_distribution(df)
        print(f"Plots saved to {self.output_dir}")

    def plot_profit_composition(self, df: pd.DataFrame):
        """
        [#47] 利润构成分析：堆叠柱状图
        展示 Net Profit 是如何由 Revenue 和 Cost 组成的。
        """
        plt.figure(figsize=(14, 7))

        # 准备数据
        # Revenue > 0
        revenue_cols = ['revenue_energy', 'revenue_reg']
        # Cost < 0 (为了堆叠，我们取负值画在 x 轴下方，或者正值堆叠显示)
        # 这里采用 "Waterfall" 风格或 分组堆叠
        # 简单起见：画两根柱子，左边是收入(堆叠)，右边是成本(堆叠)，中间是净利润

        # 由于我们是 Time Series (Episode)，我们画平均值的构成
        means = df.mean()

        # 收入项
        rev_vals = [means.get('revenue_energy', 0), means.get('revenue_reg', 0)]
        rev_labels = ['Energy Revenue', 'Regulation Revenue']

        # 成本项 (取绝对值画图)
        cost_vals = [
            abs(means.get('cost_aging', 0)),
            abs(means.get('cost_penalty', 0)),
            abs(means.get('cost_aux', 0))
        ]
        cost_labels = ['Aging Cost', 'Penalty Cost', 'Auxiliary Cost']

        # 净利润
        net_profit = means.get('profit_net', 0)

        # 绘图
        x = [0, 1, 2]
        x_labels = ['Total Revenue', 'Total Cost', 'Net Profit']

        # Revenue Stack
        plt.bar(0, rev_vals[0], label=rev_labels[0], color='forestgreen', alpha=0.7)
        plt.bar(0, rev_vals[1], bottom=rev_vals[0], label=rev_labels[1], color='limegreen', alpha=0.7)

        # Cost Stack
        plt.bar(1, cost_vals[0], label=cost_labels[0], color='firebrick', alpha=0.7)
        plt.bar(1, cost_vals[1], bottom=cost_vals[0], label=cost_labels[1], color='red', alpha=0.7)
        plt.bar(1, cost_vals[2], bottom=sum(cost_vals[:2]), label=cost_labels[2], color='lightcoral', alpha=0.7)

        # Net Profit
        color = 'blue' if net_profit > 0 else 'black'
        plt.bar(2, net_profit, label='Net Profit', color=color, alpha=0.7)

        plt.title(f"Average Profit Composition (Over {len(df)} Episodes)")
        plt.ylabel("CNY")
        plt.xticks(x, x_labels)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        plt.savefig(os.path.join(self.output_dir, "plot_profit_composition.png"), dpi=300)
        plt.close()

    def plot_cost_breakdown(self, df: pd.DataFrame):
        """
        成本结构饼图：看看到底谁是吞噬利润的黑洞
        """
        plt.figure(figsize=(8, 8))

        means = df.mean()
        costs = {
            'Aging': abs(means.get('cost_aging', 0)),
            'Penalty': abs(means.get('cost_penalty', 0)),
            'Auxiliary': abs(means.get('cost_aux', 0))
        }

        # 过滤掉 0 的项
        labels = []
        sizes = []
        for k, v in costs.items():
            if v > 1e-3:
                labels.append(k)
                sizes.append(v)

        if not sizes:
            return  # 没有成本数据

        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff', '#99ff99'])
        plt.title("Cost Breakdown")

        plt.savefig(os.path.join(self.output_dir, "plot_cost_breakdown.png"), dpi=300)
        plt.close()

    def plot_soc_distribution(self, df: pd.DataFrame):
        """
        绘制最终 SoC 分布，检查是否总是满电或空电 (策略模式)
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df['final_soc'], bins=20, kde=True)
        plt.title("Distribution of Final SoC (End of Episode)")
        plt.xlabel("SoC")
        plt.ylabel("Frequency")
        plt.axvline(0.5, color='r', linestyle='--', label='Target (0.5)')
        plt.legend()

        plt.savefig(os.path.join(self.output_dir, "plot_soc_dist.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    # 测试代码
    # 假设当前目录下有 eval_results
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "./eval_results"

    if os.path.exists(path):
        plotter = MarketPlotter(path)
        plotter.plot_all()
    else:
        print(f"Usage: python plotter.py <eval_result_dir>")