import mimetypes
import logging
from pathlib import Path

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """配置日志记录器"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)

def is_text_file(file_path: Path) -> bool:
    """判断是否是文本文件"""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    # 添加对csv文件类型的支持
    if file_path.suffix.lower() == '.csv':
        return True
    return mime_type is None or mime_type.startswith('text/')

def get_relative_path(path: Path, base_path: Path) -> str:
    """获取相对路径"""
    try:
        return str(path.relative_to(base_path))
    except ValueError:
        return str(path)

def create_directory_index(base_path: Path, level: int = 0) -> str:
    """创建目录索引"""
    result = []
    indent = "  " * level
    
    try:
        for item in sorted(base_path.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
            if item.is_dir():
                result.append(f"{indent}- 📁 {item.name}/")
                result.extend(create_directory_index(item, level + 1))
            else:
                result.append(f"{indent}- 📄 {item.name}")
    except Exception as e:
        logging.error(f"创建目录索引时出错: {str(e)}")
    
    return "\n".join(result)

def sanitize_filename(filename: str) -> str:
    """清理文件名中的非法字符"""
    return "".join(c for c in filename if c.isalnum() or c in "._- ")
