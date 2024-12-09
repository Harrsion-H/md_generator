from dataclasses import dataclass
from typing import Set, List, Pattern


@dataclass
class FilterConfig:
    """文件过滤配置类"""

    exclude_extensions: Set[str] = None  # 要排除的文件扩展名
    exclude_directories: Set[str] = None  # 要排除的目录名
    exclude_patterns: List[Pattern] = None  # 要排除的正则表达式模式
    min_size: int = None  # 最小文件大小(字节)
    max_size: int = None  # 最大文件大小(字节)
    include_extensions: Set[str] = None  # 只包含的文件扩展名

    def __post_init__(self):
        """初始化后处理"""
        # 确保所有集合都被初始化
        if self.exclude_extensions is None:
            self.exclude_extensions = set()
        if self.exclude_directories is None:
            self.exclude_directories = set()
        if self.exclude_patterns is None:
            self.exclude_patterns = []
        if self.include_extensions is None:
            self.include_extensions = set()

        # 统一转换为小写
        self.exclude_extensions = {ext.lower() for ext in self.exclude_extensions}
        self.include_extensions = {ext.lower() for ext in self.include_extensions}


@dataclass
class FormatConfig:
    """格式化配置类"""

    max_content_length: int = 1000000  # 最大内容长度（字节）
    max_preview_lines: int = 100  # 最大预览行数
    show_binary_preview: bool = False  # 是否显示二进制文件预览
    indent_size: int = 2  # 缩进大小
