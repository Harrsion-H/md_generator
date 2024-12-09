import re
from pathlib import Path
from src.config import FilterConfig, FormatConfig
from src.generator import MarkdownGenerator
from src.gui import launch_gui


def main():
    """主函数"""
    # 启动GUI界面
    launch_gui()


if __name__ == "__main__":
    main()
