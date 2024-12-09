import json
import xml.dom.minidom
import yaml
import csv
import io
from pathlib import Path
from typing import Dict, Callable
from .config import FormatConfig


class ContentFormatter:
    """内容格式化处理类"""

    def __init__(self, format_config: FormatConfig):
        """
        初始化格式化器
        :param format_config: 格式化配置
        """
        self.config = format_config
        self.setup_formatters()

    def setup_formatters(self):
        """设置不同类型文件的格式化器"""
        self.formatters: Dict[str, Callable] = {
            ".json": self.format_json,
            ".xml": self.format_xml,
            ".yaml": self.format_yaml,
            ".yml": self.format_yaml,
            ".csv": self.format_csv,
            ".md": self.format_markdown,
            ".txt": self.format_text,
        }

    def format_content(self, content: str, file_path: Path) -> str:
        """
        根据文件类型选择合适的格式化方法
        :param content: 文件内容
        :param file_path: 文件路径
        :return: 格式化后的内容
        """
        try:
            # 检查内容长度
            if len(content) > self.config.max_content_length:
                content = (
                    content[: self.config.max_content_length] + "\n... (内容已截断)"
                )

            # 获取文件扩展名
            ext = file_path.suffix.lower()

            # 获取对应的格式化器，如果没有特定的格式化器，则按普通代码处理
            formatter = self.formatters.get(ext, self.format_code)
            formatted_content = formatter(content, file_path)

            return self.post_process(formatted_content)
        except Exception as e:
            return f"[格式化错误: {str(e)}]\n{content}"

    def post_process(self, content: str) -> str:
        """
        后处理：处理行数限制
        :param content: 格式化后的内容
        :return: 处理后的内容
        """
        # 限制行数
        if self.config.max_preview_lines > 0:
            lines = content.split("\n")
            if len(lines) > self.config.max_preview_lines:
                content = (
                    "\n".join(lines[: self.config.max_preview_lines])
                    + "\n... (超出预览行数限制)"
                )

        return content

    def format_code(self, content: str, file_path: Path) -> str:
        """
        代码文件格式化
        :param content: 文件内容
        :param file_path: 文件路径
        :return: 格式化后的代码块
        """
        # 获取文件扩展名（去掉点号）
        ext = file_path.suffix[1:] if file_path.suffix else "text"
        return f"```{ext}\n{content}\n```"

    def format_json(self, content: str, file_path: Path) -> str:
        """JSON格式化"""
        try:
            parsed = json.loads(content)
            formatted = json.dumps(
                parsed, indent=self.config.indent_size, ensure_ascii=False
            )
            return f"```json\n{formatted}\n```"
        except:
            return self.format_code(content, file_path)

    def format_xml(self, content: str, file_path: Path) -> str:
        """XML格式化"""
        try:
            dom = xml.dom.minidom.parseString(content)
            formatted = dom.toprettyxml(indent=" " * self.config.indent_size)
            return f"```xml\n{formatted}\n```"
        except:
            return self.format_code(content, file_path)

    def format_yaml(self, content: str, file_path: Path) -> str:
        """YAML格式化"""
        try:
            parsed = yaml.safe_load(content)
            formatted = yaml.dump(
                parsed,
                allow_unicode=True,
                default_flow_style=False,
                indent=self.config.indent_size,
            )
            return f"```yaml\n{formatted}\n```"
        except:
            return self.format_code(content, file_path)

    def format_csv(self, content: str, file_path: Path) -> str:
        """CSV格式化"""
        try:
            reader = csv.reader(io.StringIO(content))
            formatted_content = "\n".join([",".join(row) for row in reader])
            return f"```csv\n{formatted_content}\n```"
        except:
            return self.format_code(content, file_path)

    def format_markdown(self, content: str, file_path: Path) -> str:
        """Markdown格式化 - 直接返回原内容"""
        return f"```md\n{content}\n```"

    def format_text(self, content: str, file_path: Path) -> str:
        """文本格式化 - 使用普通代码块"""
        return f"```text\n{content}\n```"
