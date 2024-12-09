import os
from pathlib import Path
import logging
from typing import Optional, TextIO
import mimetypes
from .config import FilterConfig, FormatConfig
from .formatter import ContentFormatter
from .utils import setup_logging, is_text_file


class MarkdownGenerator:
    def __init__(
        self,
        start_path: str,
        output_file: str,
        filter_config: Optional[FilterConfig] = None,
        format_config: Optional[FormatConfig] = None,
    ):
        self.start_path = Path(start_path)
        self.output_file = output_file
        self.filter_config = filter_config or FilterConfig()
        self.format_config = format_config or FormatConfig()
        self.formatter = ContentFormatter(self.format_config)
        self.logger = setup_logging(__name__)

    def should_process_file(self, file_path: Path) -> bool:
        """检查文件是否应该被处理"""
        try:
            file_ext = file_path.suffix.lower()

            if (
                self.filter_config.include_extensions
                and file_ext not in self.filter_config.include_extensions
            ):
                return False

            if file_ext in self.filter_config.exclude_extensions:
                return False

            for pattern in self.filter_config.exclude_patterns:
                if pattern.search(file_path.name):
                    return False

            if (
                self.filter_config.min_size is not None
                or self.filter_config.max_size is not None
            ):
                file_size = file_path.stat().st_size
                if (
                    self.filter_config.min_size is not None
                    and file_size < self.filter_config.min_size
                ):
                    return False
                if (
                    self.filter_config.max_size is not None
                    and file_size > self.filter_config.max_size
                ):
                    return False

            return True
        except Exception as e:
            self.logger.error(f"检查文件过滤条件时出错 {file_path}: {str(e)}")
            return False

    def should_process_directory(self, dir_path: Path) -> bool:
        """检查目录是否应该被处理"""
        return dir_path.name not in self.filter_config.exclude_directories

    def read_file_content(self, file_path: Path) -> str:
        """读取并格式化文件内容"""
        try:
            if not is_text_file(file_path):
                if self.format_config.show_binary_preview:
                    return "[二进制文件] - 略过内容显示"
                return ""

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return self.formatter.format_content(content, file_path)
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    content = f.read()
                    return self.formatter.format_content(content, file_path)
            except Exception as e:
                self.logger.error(f"无法读取文件 {file_path}: {str(e)}")
                return f"[Error: 无法读取文件内容 - {str(e)}]"
        except Exception as e:
            self.logger.error(f"读取文件出错 {file_path}: {str(e)}")
            return f"[Error: {str(e)}]"

    def write_to_markdown(self, content: str, file_handle: TextIO):
        """写入内容到markdown文件"""
        try:
            file_handle.write(content + "\n\n")
        except Exception as e:
            self.logger.error(f"写入内容失败: {str(e)}")

    def process_directory(self, current_path: Path, level: int, file_handle: TextIO):
        """处理目录"""
        try:
            # 先处理所有文件夹
            directories = sorted([d for d in current_path.iterdir() if d.is_dir()])
            for dir_item in directories:
                if self.should_process_directory(dir_item):
                    title_marks = "#" * level
                    self.write_to_markdown(
                        f"{title_marks} {dir_item.name}", file_handle
                    )
                    self.process_directory(dir_item, level + 1, file_handle)

            # 再处理所有文件
            files = sorted([f for f in current_path.iterdir() if f.is_file()])
            for file_item in files:
                if self.should_process_file(file_item):
                    title_marks = "#" * level
                    self.write_to_markdown(
                        f"{title_marks} {file_item.name}", file_handle
                    )
                    content = self.read_file_content(file_item)
                    if content.strip():
                        self.write_to_markdown(content, file_handle)

        except Exception as e:
            self.logger.error(f"处理目录出错 {current_path}: {str(e)}")

    def generate(self):
        """生成markdown文档"""
        try:
            self.logger.info(f"开始处理目录: {self.start_path}")

            with open(self.output_file, "w", encoding="utf-8") as f:
                # 写入起始目录作为一级标题
                self.write_to_markdown(f"# {self.start_path.name}", f)
                # 从二级标题开始处理目录内容
                self.process_directory(self.start_path, 2, f)

            self.logger.info(f"markdown文档已生成: {self.output_file}")

        except Exception as e:
            self.logger.error(f"生成markdown文档失败: {str(e)}")
