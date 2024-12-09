import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
from pathlib import Path
from .config import FilterConfig, FormatConfig
from .generator import MarkdownGenerator

class MarkdownGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Markdown Generator")
        self.root.geometry("800x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Directory Selection
        ttk.Label(self.main_frame, text="源目录:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.source_dir = tk.StringVar()
        ttk.Entry(self.main_frame, textvariable=self.source_dir, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(self.main_frame, text="浏览", command=self.browse_source).grid(row=0, column=2, padx=5, pady=5)

        # Output File Selection
        ttk.Label(self.main_frame, text="输出文件:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_file = tk.StringVar(value="output.md")
        ttk.Entry(self.main_frame, textvariable=self.output_file, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(self.main_frame, text="浏览", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        # Filter Settings Frame
        filter_frame = ttk.LabelFrame(self.main_frame, text="过滤设置", padding="5")
        filter_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # Include Extensions
        ttk.Label(filter_frame, text="包含的扩展名:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.include_extensions = tk.StringVar(value=".py,.txt,.md,.json,.yaml,.yml,.csv")
        ttk.Entry(filter_frame, textvariable=self.include_extensions, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)

        # Exclude Extensions
        ttk.Label(filter_frame, text="排除的扩展名:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.exclude_extensions = tk.StringVar(value=".pyc,.git,.pdf,.jpg,.png")
        ttk.Entry(filter_frame, textvariable=self.exclude_extensions, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Exclude Directories
        ttk.Label(filter_frame, text="排除的目录:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.exclude_directories = tk.StringVar(value=".git,__pycache__,node_modules,.idea")
        ttk.Entry(filter_frame, textvariable=self.exclude_directories, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)

        # Size Limits
        size_frame = ttk.Frame(filter_frame)
        size_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(size_frame, text="最小文件大小 (bytes):").grid(row=0, column=0, sticky=tk.W)
        self.min_size = tk.StringVar(value="0")
        ttk.Entry(size_frame, textvariable=self.min_size, width=15).grid(row=0, column=1, padx=5)
        
        ttk.Label(size_frame, text="最大文件大小 (bytes):").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.max_size = tk.StringVar(value="1048576")  # 1MB
        ttk.Entry(size_frame, textvariable=self.max_size, width=15).grid(row=0, column=3, padx=5)

        # Format Settings Frame
        format_frame = ttk.LabelFrame(self.main_frame, text="格式设置", padding="5")
        format_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(format_frame, text="最大内容长度:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.max_content_length = tk.StringVar(value="100000000")
        ttk.Entry(format_frame, textvariable=self.max_content_length, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Label(format_frame, text="最大预览行数:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.max_preview_lines = tk.StringVar(value="10000")
        ttk.Entry(format_frame, textvariable=self.max_preview_lines, width=20).grid(row=1, column=1, sticky=tk.W, pady=5)

        self.show_binary_preview = tk.BooleanVar(value=False)
        ttk.Checkbutton(format_frame, text="显示二进制文件预览", variable=self.show_binary_preview).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Generate Button
        ttk.Button(self.main_frame, text="生成Markdown", command=self.generate_markdown).grid(row=4, column=0, columnspan=3, pady=20)

    def browse_source(self):
        directory = filedialog.askdirectory()
        if directory:
            self.source_dir.set(directory)

    def browse_output(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")]
        )
        if file_path:
            self.output_file.set(file_path)

    def str_to_set(self, value):
        return {x.strip() for x in value.split(",") if x.strip()}

    def generate_markdown(self):
        try:
            # Create filter config
            filter_config = FilterConfig(
                exclude_extensions=self.str_to_set(self.exclude_extensions.get()),
                exclude_directories=self.str_to_set(self.exclude_directories.get()),
                exclude_patterns=[
                    re.compile(r".*\.temp$"),
                    re.compile(r".*\.bak$"),
                    re.compile(r".*~$"),
                ],
                min_size=int(self.min_size.get()),
                max_size=int(self.max_size.get()),
                include_extensions=self.str_to_set(self.include_extensions.get()),
            )

            # Create format config
            format_config = FormatConfig(
                max_content_length=int(self.max_content_length.get()),
                max_preview_lines=int(self.max_preview_lines.get()),
                show_binary_preview=self.show_binary_preview.get(),
                indent_size=2,
            )

            # Create generator instance
            generator = MarkdownGenerator(
                start_path=self.source_dir.get(),
                output_file=self.output_file.get(),
                filter_config=filter_config,
                format_config=format_config,
            )

            # Generate markdown
            generator.generate()
            messagebox.showinfo("成功", "Markdown文件生成完成！")
            
        except Exception as e:
            messagebox.showerror("错误", f"生成过程中发生错误：{str(e)}")

def launch_gui():
    root = tk.Tk()
    app = MarkdownGeneratorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
